import os
import csv
import numpy as np
import pandas as pd
import torch.utils.data as data_utils
from PIL import Image
from vocparseclslabels import PascalVOC


class VOC2012ClassificationDataset(data_utils.Dataset):
    """
    Dataset for paths to image and label
    Args:
        root: Root Directory
        train_or_val: 'train' or 'val' for dataset type in txt files

    Creates a dataset: list of [image tensors, one-hot encoding] for every image in 'train' or 'val'
    """
    def __init__(self, root, train_or_val, transform=None, target_transform=None):
        self.root = root
        self.pv = PascalVOC(os.path.join(root, 'VOCdevkit', 'VOC2012'))
        self.image_dir = self.pv.img_dir
        self.set_dir = self.pv.set_dir
        self.train_or_val = train_or_val

        self.classes = self.pv.list_image_sets()
        self.class_to_index = {}
        self.index_to_class = {}
        # For one hot indexing
        for i, cat_name in enumerate(self.classes):
            self.class_to_index[cat_name] = i
            self.index_to_class[i] = cat_name
        
        print("Class to index mapping:\n", self.class_to_index)

        # k, v: image_name, one-hot category of len classes
        self.images = {}
        # Store image, labels
        for cat_name, idx in self.class_to_index.items():
            img_ls = self.pv.imgs_from_category_as_list(cat_name, train_or_val)
            # Iterate through list of images in a category
            for img_name in img_ls:
                # One-hot encode images
                if img_name in self.images:
                    self.images[img_name][idx] = 1
                else:
                    self.images[img_name] = np.zeros(len(self.classes))
                    self.images[img_name][idx] = 1

        # TODO: Change this hacky solution, converted dict to list for enum in Dataloader
        self.image_list = []
        for k, v in self.images.items():
            temp = [k,v]
            self.image_list.append(temp)

        self.transform = transform
        self.target_transform = target_transform

        
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, index):
        """
        Returns:
            image as PIL.Image
            target as one-hot encode of class, np.array of length num_classes
        """
        img_name, target = self.image_list[index]
        image_path = os.path.join(self.image_dir, img_name + '.jpg')
        # Get image and transform image and label
        image = Image.open(image_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target
