import os
import csv
import numpy as np
import pandas as pd
import torch.utils.data as data_utils
from PIL import Image

classes = ['aeroplane', 'bicycle', 'bird', 'boat',
                     'bottle', 'bus', 'car', 'cat', 'chair',
                     'cow', 'diningtable', 'dog', 'horse',
                     'motorbike', 'person', 'pottedplant',
                     'sheep', 'sofa', 'train', 'tvmonitor']


def read_image_label(file):
    """
    Read image and label from txt file
    Returns:
        data: dictionary {img_name: label}
    """
    data = {}
    with open(file, 'r') as f:
        for line in f:
            line = line.split()
            img_name = line[0]
            label = int(line[-1])
            data[img_name] = label
    return data


def read_object_labels(devkit_path, dataset, train_or_val):
    """
    
    """
    # All train, val and trainval txt files in Main
    path_labels = os.path.join(devkit_path, dataset, 'ImageSets', 'Main')
    labelled_data = {}
    num_classes = len(classes)
    
    for i in range():
        file = os.path.join(path_labels, classes[i] + '_' + train_or_val + '.txt')
        data = read_image_label(file)

        if i == 0:
            for img_name, label in data.items():
                labels = np.zeros(num_classes)
                labels[i] = label
                labelled_data[img_name] = labels
        else:
            for img_name, label in data.items():
                labelled_data[img_name][i] = label
    
    return labelled_data


def write_object_labels_csv(file, labelled_data):
    # write a csv file
    with open(file, 'w') as csvfile:
        fieldnames = ['name']
        fieldnames.extend(classes)
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for (name, labels) in labelled_data.items():
            example = {'name': name}
            for i in range(20):
                example[fieldnames[i + 1]] = int(labels[i])
            writer.writerow(example)

    csvfile.close()


class VOC2012ClassificationDataset(data_utils.Dataset):
    """
    Dataset for paths to image and label
    """
    def __init__(self, root, train_or_val, transform=None, target_transform=None):
        self.root = root
        self.devkit_path = os.path.join(root, 'VOCdevkit')
        self.image_path = os.path.join(self.devkit_path, 'VOC2012', 'JPEGImages')
        self.train_or_val = train_or_val
        self.transform = transform
        self.target_transform = target_transform

        path_csv = os.path.join(self.root, 'files', 'VOC2012')
        file_csv = os.path.join(path_csv, 'classification_' + train_or_val + '.csv')

        # Create csv file
        if not os.path.exists(file_csv):
            if not os.path.exists(path_csv):
                os.makedirs(path_csv)
            labelled_data = read_object_labels(self.devkit_path, 'VOC2012', self.train_or_val)



    def __len__(self):
        return len(self.label_list)
    
    def __getitem__(self, index):
        # A 2-el list, [image, label]
        img_label = self.label_list[index]

        image_path = os.path.join(self.image_dir, img_label[0])
        
        label = int(img_label[1])

        # Get image and transform image and label
        image = self.transform(Image.open(image_path).convert('RGB'))

        return image, label

