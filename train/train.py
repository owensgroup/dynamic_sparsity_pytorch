import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

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
    dataset='CIFAR10',
    model='Ampere_Resnet18',
    logging_interval=4)


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

def train(model, optimizer, loss_fn, train_loader: DataLoader , val_loader, test_loader, cfg):
  batches_per_epoch = len(train_loader) / cfg.batch_size

  
  log_ckpt = batches_per_epoch / cfg.logging_interval
  step = 0
  model = model.train()
  for epoch in range(cfg.epochs):
    running_loss = 0.0
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
      step += 1
      if i % log_ckpt == log_ckpt - 1:
        val_acc = calc_accuracy(val_loader, model)
        print('[%d, %5d] loss: %.3f, validation accuracy: %.2f' %
                    (epoch + 1, i + 1, running_loss / log_ckpt, val_acc))
        wandb.log({"loss": running_loss / log_ckpt, 'Validation Accuracy (%)': val_acc}, step=step)
    test_acc = calc_accuracy(test_loader, model)
    wandb.log({'Test Accuracy (%)': test_acc})
    print('[%d / %d] Test Accuracy : %.2f' % (epoch+1, cfg.epochs, test_acc))

def run(cfg: wandb.config):
  model, optimizer, loss_fn, train_loader, val_loader, test_loader = make(cfg)
  train(model, optimizer, loss_fn, train_loader, val_loader, test_loader, cfg)

wandb.init(config=hyperparameter_defaults)
config = wandb.config
run(config)
