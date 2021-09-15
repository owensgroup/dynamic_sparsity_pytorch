from typing import Tuple
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets
from torchvision.models import resnet18, wide_resnet50_2, resnet50
import torchvision.transforms as transforms

import wandb

from models.sparse_resnet import sparse_resnet18, sparse_resnet50, sparse_wide_resnet50_2
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

hyperparameter_defaults = dict(
    epochs=100,
    batch_size=64,
    learning_rate=.001,
    val_ratio=.2,
    logging_rate=100,
    sparse_monitor=True,
    dataset='CIFAR10',
    model='Resnet18')


def make_dataset(cfg: wandb.config):
  transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
  if cfg.dataset == 'CIFAR10':
    dset = datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
    testset = datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
  elif cfg.dataset == 'CIFAR100':
    dset = datasets.CIFAR100(root='./data', train=True,
                                        download=True, transform=transform)
    testset = datasets.CIFAR100(root='./data', train=False,
                                       download=True, transform=transform)
  else:
    raise ValueError('No supported datasets selected')

  trainset, valset = torch.utils.data.random_split(dataset=dset,lengths=[
                                                int((1-cfg.val_ratio) * len(dset)),
                                                int(cfg.val_ratio * len(dset))
                                                ])
  trainloader = torch.utils.data.DataLoader(trainset, batch_size=cfg.batch_size,
                                          shuffle=True, pin_memory=True, num_workers=8)
  valloader = torch.utils.data.DataLoader(valset, batch_size=cfg.batch_size,
                                        shuffle=True, pin_memory=True, num_workers=8)
  testloader = torch.utils.data.DataLoader(testset, batch_size=cfg.batch_size,
                                         shuffle=False, num_workers=8)
  return trainloader, valloader, testloader

def make_model(cfg: wandb.config):
  if cfg.model == 'Ampere_Resnet18':
    net = sparse_resnet18()
  elif cfg.model == 'Ampere_Resnet50':
    net = sparse_resnet50()
  elif cfg.model == 'Ampere_WideResnet50':
    net = sparse_wide_resnet50_2()
  elif cfg.model == 'Resnet18':
    net = resnet18()
  elif cfg.model == 'Resnet50':
    net = resnet50()
  elif cfg.model == 'WideResnet50':
    net = wide_resnet50_2()
  net = net.to(device)
  optimizer = optim.Adam(net.parameters(), lr=cfg.learning_rate)
  criterion = nn.CrossEntropyLoss()
  return net, optimizer, criterion

def make(config: wandb.config):
  train_loader, val_loader, test_loader = make_dataset(config)
  model, optimizer, loss_fn = make_model(config)
  return model, optimizer, loss_fn, train_loader, val_loader, test_loader

def calc_accuracy(loader, net):
  correct = 0
  total = 0
  with torch.no_grad():
    for data in loader:
      images, labels = data
      images = images.to(device)
      labels = labels.to(device)
      raw_scores = net(images)
      _, predicted = torch.max(raw_scores.data,1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()
  return 100 * correct/total

# Hooks
class SparseMonitorHook():
  def __init__(self):
    self.sparse_level = 0.0
  
  def hook(self, model: nn.Module, input: torch.Tensor, output: torch.Tensor):
    self.sparse_level = torch.count_nonzero(input[0]) / input[0].numel()


class ModelHook(nn.Module):
  def __init__(self, model: nn.Module, sparse_monitor: bool, log_int: int):
    super().__init__()
    self.model = model
    self.steps = 0
    self.log_int = log_int
    self.hooks = {}
    self.sparse_monitor = sparse_monitor
    if sparse_monitor:
      print('Registering Sparse Monitor Hooks...')
      for name, layer in self.model.named_modules():
        if isinstance(layer, (nn.Linear, nn.Conv2d)):
          self.hooks[name] = SparseMonitorHook()
          layer.register_forward_hook(self.hooks[name].hook)


  def forward(self, x: torch.Tensor):
    out = self.model(x)
    # Log to WandB
    if self.sparse_monitor and (self.steps % self.log_int == 0):
      print('Logging Per Layer Activation Sparsity...')
      for name, hook in self.hooks.items():
        wandb.log({f'{name} % (NNZ/NUM_EL)': hook.sparse_level}, step=self.steps)
        print(f'{name}: {hook.sparse_level}')
    self.steps += 1
    return out
  

def train(model, optimizer, loss_fn, train_loader, val_loader, test_loader, cfg):
  batch_count = len(train_loader)
  step = 0
  model = model.train()
  model = ModelHook(model, cfg.sparse_monitor, cfg.logging_rate)
  print('Starting Training')
  for epoch in range(cfg.epochs):
    running_loss = 0.0
    print('Running Epoch')
    for i, data in enumerate(train_loader, 0):
      inputs, labels = data
      inputs = inputs.to(device)
      labels = labels.to(device)
      optimizer.zero_grad()
      raw_scores = model(inputs)
      loss = loss_fn(raw_scores, labels)
      loss.backward()
      optimizer.step()
      running_loss += loss.item()
      if model.steps % cfg.logging_rate == 0:
        val_acc = calc_accuracy(val_loader, model)
        print('[%d, %5d] loss: %.3f, validation accuracy: %.2f' %
                    (i+1, batch_count, running_loss / i, val_acc))
        wandb.log({"loss": running_loss / i, 'Validation Accuracy (%)': val_acc}, step=model.steps)
    test_acc = calc_accuracy(test_loader, model)
    wandb.log({'Test Accuracy (%)': test_acc})
    print('[%d / %d] Test Accuracy : %.2f' % (epoch+1, cfg.epochs, test_acc))

def run(cfg: wandb.config):
  model, optimizer, loss_fn, train_loader, val_loader, test_loader = make(cfg)
  train(model, optimizer, loss_fn, train_loader, val_loader, test_loader, cfg)

wandb.init(config=hyperparameter_defaults)
config = wandb.config
run(config)
