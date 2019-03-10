import numpy as np
import random
import matplotlib.pyplot as plt
import dataset as ds
import torchvision.transforms as transforms
import torch.utils.data as data_utils

root = './'
batch_size = 1

def show_dataset(dataset, n=6):
    """
    Plots n random images with corresponding labels
    """
    fig = plt.figure()
    for i in range(n):
        plt.subplot(2, 3, i+1)
        rand_idx = random.randint(0, len(dataset))
        plt.imshow(dataset[rand_idx][0])
        labels = []
        for idx, truth in enumerate(dataset[rand_idx][1]):
            if truth == 1:
                labels.append(dataset.index_to_class[idx])
        plt.title(', '.join(labels))
    plt.show()

def test():
    transform = transforms.Compose([
        transforms.Resize(224)
    ])
    
    train_dataset = ds.VOC2012ClassificationDataset(root, 'train', transform=transform)
    val_dataset = ds.VOC2012ClassificationDataset(root, 'val', transform=transform)
    print("Length train dataset:", len(train_dataset))
    print("Length validation dataset:", len(val_dataset))

    show_dataset(train_dataset)


if __name__=='__main__':
    test()