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
from PIL import Image

from models import dataset

class Classifier(object):
    def __init__(self, model_path, use_cuda=True):
        torch.manual_seed(1)
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.model = self._load_model(model_path)
        self.model = self.model.to(self.device)
        self.model.eval()

    def _list_image_sets(self):
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

    def index_to_class(self):
        index_to_class = {}
        for i, cat_name in enumerate(self._list_image_sets()):
            index_to_class[i] = cat_name
        return index_to_class

    def class_to_index(self):
        class_to_index = {}
        for i, cat_name in enumerate(self._list_image_sets()):
            class_to_index[cat_name] = i
        return class_to_index
    
    def predict(self, image_path, display_threshold=0.5):
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform_image(image).to(self.device)
        pred = self.model(image_tensor)
        pred_avg = torch.sigmoid(pred.view(1, 5, -1).mean(1))

        index_to_class = self.index_to_class()
        labels = []
        for idx, prob in enumerate(pred_avg.tolist()[0]):
            if prob > display_threshold:
                labels.append(index_to_class[idx] + ' {:.1f}%'.format(prob*100))

        # if not labels:
        #     prob = max(pred_avg.tolist()[0])
        #     labels.append(index_to_class[idx] + ' {:.1f}%'.format(prob*100))

        return labels

    def _load_model(self, path):
        model = models.resnet18(pretrained=False, num_classes=20)
        model.load_state_dict(torch.load(path))
        return model

    def transform_image(self, image):
        test_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.FiveCrop(224),
                transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])), 
                transforms.Lambda(lambda crops: torch.stack([transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(crop) for crop in crops])), 
                ])
        return test_transform(image)

# from app.classifier import Classifier as Clf
# clf = Clf('./results/pascalvoc_A.pt')
# clf.predict('2007_002120.jpg')