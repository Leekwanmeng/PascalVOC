import numpy as np, os, sys
import time
import argparse
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import torchvision.models as models
import torch.optim
import torchnet.meter
from PIL import Image
import utils
from train_model import load_model, initialise_transforms
from dataset import VOC2012ClassificationDataset as VOCDataset
from sklearn.metrics import accuracy_score

root = './'


def display_prediction(pred, image, display_threshold=0.5):
    index_to_class = utils.index_to_class()
    labels = []
    for idx, prob in enumerate(pred.tolist()[0]):
        if prob > display_threshold:
            labels.append(index_to_class[idx] + ' {:.1f}%'.format(prob*100))
    
    # Display image with labels
    plt.imshow(image)
    plt.title(', '.join(labels))
    plt.axis('off')
    plt.show()
    return


def run():
    parser = argparse.ArgumentParser(description='Pascal VOC 2012 Classifier')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--test-batch-size', type=int, default=32, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--mode', type=str, default='A', metavar='M',
                        help='Mode of model')
    parser.add_argument('--demo_mode', type=str, default='single', metavar='M',
                        help='Mode of demo')
    parser.add_argument('--image_path', type=str, default='./test.jpg', metavar='M',
                        help='Mode of demo')
    # parser.add_argument('--class_name', type=str, default='aeroplane', metavar='M',
    #                     help='Mode of demo')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    # Get transform
    _, test_transform = initialise_transforms()

    # Initialise model
    model, params = load_model()
    model = model.to(device)
    model.eval()
    model_name = 'pascalvoc_' + args.mode + '.pt'
    print('Loading model...')
    model.load_state_dict(torch.load(model_name))
    
    # Convert jpg to tensor
    if args.demo_mode == 'single':
        image = Image.open(args.image_path).convert('RGB')
        image_tensor = test_transform(image).unsqueeze(0).to(device)
        # Get model prediction
        pred = model(image_tensor)
        pred = F.sigmoid(pred)
        display_prediction(pred, image)

    elif args.demo_mode == 'gui':
        class_to_index = utils.class_to_index()
        # index = class_to_index[args.class_name]

        # 2-part transform to preserve image after first_transform
        first_transform = transforms.Compose([
                transforms.Resize(224)
                ])
        second_transform = transforms.Compose([
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])

        # Validation set
        test_loader = torch.utils.data.DataLoader(
            VOCDataset(root, 'val', transform = test_transform),
            batch_size=args.test_batch_size, shuffle=True)
        
        # Get predictions on validation set
        model.eval()
        all_predictions = []
        start = time.time()
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                output = F.sigmoid(output)
                target = target.float()
                # Precision for each class in each example
                for i in range(output.shape[0]):
                    example_predictions = []
                    scores = target[i]*output[i] # Ground truth as mask for predictions
                    all_predictions.append(scores)
        
        end = time.time()
        print("Time lapsed: {:.2f}s".format((end - start)))
        print(all_predictions)
        
    else:
        raise Exception("Please enter demo_mode as 'single' or 'gui'")
    
    

if __name__ == '__main__':
    # python demo.py --demo_mode single --image_path ./2007_002120.jpg
    # python demo.py --demo_mode gui --class_name aeroplane --top 5
    run()