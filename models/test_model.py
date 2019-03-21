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
import PIL
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


def load_model(path):

    model = models.resnet18(pretrained=False, num_classes=20)

    model.load_state_dict(torch.load(path))
    
    return model


def test(args, model, device, test_loader, lossfunction):
    #Set model to testing mode
    model.eval()
    test_loss = 0
    apmeter = torchnet.meter.APMeter()
    results = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            #Batchsize, number of crops, channels, height, width
            bs, ncrops, c, h, w = data.size()
            
            output = model(data.view(-1, c, h, w))
            #Calculate mean loss of 5 crops
            output_avg = output.view(bs, ncrops, -1).mean(1)
            print(output_avg.shape)
            results.append(output_avg)
            target = target.float()
            
            test_loss += lossfunction(output_avg, target, reduction='sum').item() # sum up batch loss
            apmeter.add(torch.sigmoid(output_avg), target)

    test_loss /= len(test_loader.dataset)

    apvalue = apmeter.value().tolist()

    print('\nValidation set of {}: Average loss: {:.4f}, mean AP measurement: {:.2f} %\n'.format(
        len(test_loader.dataset) ,test_loss, sum(apvalue)/len(apvalue) * 100 ))
    
    for idx, ap in enumerate(apvalue):
        print('{:12}:\t{:.2f} %'.format(list_image_sets()[idx], ap * 100))

    return test_loss, apvalue, results

def run():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')

    parser.add_argument('--test-batch-size', type=int, default=32, metavar='N',
                        help='input batch size for testing (default: 1000)')

    parser.add_argument('--epochs', type=int, default=15, metavar='N',
                        help='number of epochs to train (default: 15)')
    
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    
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

    root = 'D:/Downloads/Deep Learning/Week 6'

    test_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.FiveCrop(224),
                transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])), 
                transforms.Lambda(lambda crops: torch.stack([transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(crop) for crop in crops])), 
                ])
    # test_transform = transforms.Compose([
    #             transforms.Resize(224),
    #             transforms.CenterCrop(224),
    #             transforms.ToTensor(),
    #             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    #             ])

    #Get dataset and input into Dataloader

    test_loader = torch.utils.data.DataLoader(
        VOCDataset(root, 'val', transform = test_transform),
        batch_size=args.test_batch_size, shuffle=False)

    test_loss_function = F.binary_cross_entropy_with_logits
    
    #Define Model
    model = load_model('./results/pascalvoc_A.pt')
    model = model.to(device)

    val_loss, val_acc, output = test(args, model, device, test_loader, test_loss_function)
    print(len(output))
    torch.save(output, 'val_set_results.pt')

if __name__ == "__main__":
    run()

# Validation set of 5823: Average loss: 1.6819, mean AP measurement: 81.93 %

# aeroplane   :   96.42 %
# bicycle     :   81.16 %
# bird        :   94.55 %
# boat        :   86.86 %
# bottle      :   61.46 %
# bus         :   92.09 %
# car         :   78.08 %
# cat         :   95.24 %
# chair       :   71.65 %
# cow         :   79.79 %
# diningtable :   64.22 %
# dog         :   90.64 %
# horse       :   88.48 %
# motorbike   :   87.10 %
# person      :   94.93 %
# pottedplant :   53.43 %
# sheep       :   86.34 %
# sofa        :   58.19 %
# train       :   93.88 %
# tvmonitor   :   84.03 %