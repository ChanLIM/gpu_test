from __future__ import print_function

import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.models.resnet import resnet34


def main():
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 16 * 5 * 5)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return F.log_softmax(x, dim=0)


    def loss_func(output, target):
        return F.nll_loss(output, target)

    def train(epoch):
        model.train()

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            output = model(data)
            output = F.log_softmax(output, dim=1)
            loss = loss_func(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 50 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))

    def test():
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                data, target = Variable(data), Variable(target)
                output = F.log_softmax(model(data), dim=1)
                # sum up batch loss
                test_loss += F.nll_loss(output, target, reduction='sum').data
                # get the index of the max log-probability
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))


    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')  
    parser.add_argument('--vram',type=int,default=8)
    parser.add_argument('--num_gpus',type=int,default='1',help='gpu_num')
    args = parser.parse_args()


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.DataParallel(resnet34(num_classes=10).to(device))

    optimizer = optim.Adam(model.parameters(), lr=0.01)

    os.system('nvidia-smi')
    os.system('nvidia-smi topo -m')
    os.system('sudo nvidia-smi nvlink -sc 0bz')

    # Training settings
    batch_size = 8*args.vram*args.num_gpus

    # CIFAR10 Dataset
    train_dataset = datasets.CIFAR10(root='./data/',
                                   train=True,
                                   transform=transforms.ToTensor(),
                                   download=True)

    test_dataset = datasets.CIFAR10(root='./data/',
                                  train=False,
                                  transform=transforms.ToTensor())

    # Data Loader (Input Pipeline)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=4)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=4)

    print("\nTRAINING START")
    for epoch in range(1, 20):
        print("===============================================================================")
        train(epoch)
        test()
        print("NVLINK TRAFFIC RX/TX")
        os.system("sudo nvidia-smi nvlink -g 0")
        print("===============================================================================")

    os.system("sudo nvidia-smi nvlink -sc 0bn")


if __name__ == "__main__":
    main()
