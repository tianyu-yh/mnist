import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torch.autograd import Variable
# from time import time
from model import *
import numpy
device = torch.device("cuda:0"if torch.cuda.is_available() else "cpu")



batch_size = 64

train_dataset = datasets.MNIST(root='/home/a313/ty/minist/',
                               train=True,
                               transform=transforms.ToTensor(),
                               download=True)

test_dataset = datasets.MNIST(root='/home/a313/ty/minist/',
                              train=False,
                              transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
model = LeNet5()
model = model.to(device)

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
criterion = nn.CrossEntropyLoss().to(device)


def train(model, num_epoch):
    Loss = []
    model.train(True)
    for i in range(num_epoch):
        loss_data = 0
        accuracy = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = Variable(data).to(device), Variable(target).to(device)
            output = model(data)
            pred = output.data.max(1, keepdim=True)[1]
            accuracy += pred.eq(target.data.view_as(pred)).to(device).sum()
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_data += loss.data.item()

        epoch_loss = loss_data / len(train_loader)
        epoch_acc = accuracy.data.item() / len(train_dataset)

        print("epoch {} loss_data {} loss {}".format(i, loss_data, epoch_loss))
        print("epoch {} accuracy {} acc {}".format(i, accuracy.data.item(), epoch_acc))
        Loss.append(epoch_loss)


    return Loss


def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = Variable(data).to(device), Variable(target).to(device)
        output = model(data)

        test_loss += criterion(output, target).data.item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).to(device).sum()

    test_loss /= len(test_loader)
    print("test_loss{}".format(test_loss))
    acc = correct.item() / len(test_dataset)
    print("test acc:", acc)


def main():

    train(model, 20)
    test()
    # plt.figure()
    # plt.plot(result)
    # plt.xlabel('epoch')
    # plt.ylabel('loss')
    # plt.show()




if __name__ == '__main__':
    main()





