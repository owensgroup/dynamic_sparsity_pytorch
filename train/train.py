#%%
import enum
import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
from torchvision import datasets
import torch.optim as optim
from models.sparse_resnet import sparse_resnet18, sparse_resnet50

#%%
import wandb

batch_size = 64
learning_rate = .001
num_epochs = 100
wandb.init(project='ampere_acc_test', entity='tejalapeno')
config = wandb.config
config.batch_size = batch_size
config.learning_rate = learning_rate
config.epochs = num_epochs
config.dataset = 'CIFAR10'
config.model = 'Ampere_Resnet18'
# config = dict(
#   epochs=5,
#   batch_size=64,
#   learning_rate=.001,
#   device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
#   dataset="CIFAR10",
#   model='sparse_resnet18')

# #%%
# def model_pipeline(hyperparameters):
#   with wandb.init(project='dynamic_sparsity', config=hyperparameters):
#     config = wandb.config
#     model, train_loader, test_loader, criterion, optimizer = make(config)
#     print(model)

#%% CIFAR10
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


dset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainset, valset = torch.utils.data.random_split(dataset=dset,lengths=[
                                                int(.8 * len(dset)),
                                                int(.2 * len(dset))
                                                ])
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=8)
valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                        shuffle=True, num_workers=8)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=8)



#%%
net = sparse_resnet18().to(device)
optimizer = optim.Adam(net.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()
#%%
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
#%%
#wandb.watch(net)

for epoch in range(num_epochs):
  running_loss = 0.0
  for i, data in enumerate(trainloader, 0):
    inputs, labels = data
    inputs = inputs.to(device)
    labels = labels.to(device)
    optimizer.zero_grad()
    raw_scores = net(inputs)
    loss = criterion(raw_scores, labels)
    loss.backward()
    optimizer.step()
    running_loss += loss.item()
    if i % 500 == 499:
      wandb.log({"loss": running_loss / 500})
      val_acc = calc_accuracy(valloader, net)
      print('[%d, %5d] loss: %.3f, validation accuracy: %.2f' %
                  (epoch + 1, i + 1, running_loss / 500, val_acc))
      wandb.log({'Validation Accuracy (%)': val_acc})
  test_acc = calc_accuracy(testloader, net)
  wandb.log({'Test Accuracy (%)': test_acc})



