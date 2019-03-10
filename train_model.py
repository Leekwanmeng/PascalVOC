import numpy as np, os, sys
import argparse
import matplotlib.pyplot as plt #patch-wise similarities, droi images
import torch.nn.functional as F
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import torchvision.models as models
import torch.optim
from math import ceil
import torchnet.meter

from dataset import VOC2012ClassificationDataset as VOCDataset

def list_image_sets():
        """
        Summary: 
            List all the image sets from Pascal VOC. Don't bother computing
            this on the fly, just remember it. It's faster.
        """
        return [
            'aeroplane', 'bicycle', 'bird', 'boat',
            'bottle', 'bus', 'car', 'cat', 'chair',
            'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train',
            'tvmonitor']

def load_model():

    model = models.resnet18(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    in_ftrs = model.fc.in_features
    model.fc = nn.Linear(in_ftrs, 20)
    #Reshape last layer
    #Train only last layer
    parameters = model.fc.parameters()
    
    return model, parameters

def train(args, model, device, train_loader, optimizer, epoch, lossfunction):
    #Set model to training mode
    model.train()
    total_loss = 0
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        #Batchsize, number of crops, channels, height, width   
        optimizer.zero_grad()
        pred = model(x)
        y = y.float()
        loss = lossfunction(pred, y)
        total_loss += loss
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(x), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

    return total_loss/ceil(len(train_loader.dataset)/train_loader.batch_size)


def test(args, model, device, test_loader, lossfunction):
    #Set model to testing mode
    model.eval()
    test_loss = 0
    apmeter = torchnet.meter.APMeter()

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)
            target = target.float()
            test_loss += lossfunction(output, target, reduction='sum').item() # sum up batch loss
            apmeter.add(output, target)

    test_loss /= len(test_loader.dataset)

    print('\nValidation set: Average loss: {:.4f}\nAP measurement:'.format(test_loss))
    
    for idx, ap in enumerate(apmeter.value().tolist()):
        print('{:12}:\t{:.2f} %'.format(list_image_sets()[idx], ap * 100))

    return test_loss, apmeter.value().tolist()

def run():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')

    parser.add_argument('--test-batch-size', type=int, default=32, metavar='N',
                        help='input batch size for testing (default: 1000)')

    parser.add_argument('--epochs', type=int, default=30, metavar='N',
                        help='number of epochs to train (default: 30)')
    
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--mode', type=str, default='A', metavar='M',
                        help='Mode of model')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    root = 'D:\Downloads\Deep Learning\Week 6'

    train_transform = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor()
                ])

    test_transform = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor()
                ])

    #Get dataset and input into Dataloader
    train_loader = torch.utils.data.DataLoader(
        VOCDataset(root, 'train', transform = train_transform),
        batch_size=args.batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        VOCDataset(root, 'val', transform = train_transform),
        batch_size=args.test_batch_size, shuffle=True)

    #Define Loss function
    #train_loss_function = F.binary_cross_entropy
    train_loss_function = nn.modules.BCEWithLogitsLoss()
    test_loss_function = F.binary_cross_entropy_with_logits
    
    #Define Model
    model, params = load_model()
    model = model.to(device)

    #print('\nRunning model {}, mode: {}\n'.format(model.__class__.__name__, args.mode))
    #Define Optimizer
    #optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum)
    optimizer= torch.optim.Adam(params, lr=args.lr, )
    
    best_loss = -1
    train_loss_epoch = []
    val_loss_epoch = []
    val_acc_epoch = []

    for epoch in range(1, (args.epochs + 1)):
        train_loss = train(args, model, device, train_loader, optimizer, epoch, train_loss_function)
        val_loss, val_acc = test(args, model, device, test_loader, test_loss_function)

        train_loss_epoch.append(train_loss.item())
        val_acc_epoch.append(val_acc)
        val_loss_epoch.append(val_loss)

        if best_loss < 0 or val_loss < best_loss:
            best_loss = val_loss
            best_param = model.state_dict()
            print("FOUND BETTER MODEL, SAVING WEIGHTS...\n")
    
    results = {
        "train_loss": train_loss_epoch,
        "val_loss": val_loss_epoch,
        "val_acc": val_acc_epoch
    }

    print('Saving model...')
    torch.save(best_param, 'resnet18_' + args.mode + '.pt')
    print('Model saved as : {}\n'.format('resnet18_' + args.mode + '.pt'))

    print('Saving results...')
    torch.save(results, 'resnet18_' + args.mode + '_results' + '.pt')
    print('Results saved as : {}'.format('resnet18_' + args.mode + '_results' + '.pt'))

if __name__ == '__main__':
    run()