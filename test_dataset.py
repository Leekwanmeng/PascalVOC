import dataset as ds
import torchvision.transforms as transforms
import torch.utils.data as data_utils

root = './'
batch_size = 1

def test():
    transform = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
    
    train_dataset = ds.VOC2012ClassificationDataset(root, 'train', transform=transform)
    val_dataset = ds.VOC2012ClassificationDataset(root, 'val', transform=transform)
    print(len(train_dataset))
    print(len(val_dataset))

    # Dataloaders
    train_dataloader = data_utils.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = data_utils.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    for batch_idx, (image, label) in enumerate(train_dataloader):
        print("Image:", image)
        print("Label:", label.item)
        if batch_idx >= 1: break

    for batch_idx, (image, label) in enumerate(val_dataloader):
        print("Image:", image)
        print("Label:", label)
        if batch_idx >= 1: break

if __name__=='__main__':
    test()