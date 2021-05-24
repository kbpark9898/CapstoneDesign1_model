import torch
import torch.nn as nn
import torchvision.models as models 
from torchvision.datasets import ImageFolder
import torch.utils.data as data
from torchvision import datasets, transforms
import numpy as np
import pandas as pd
from losses import FocalLoss

print(torch.cuda.is_available())
device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")
print(f'\ndevice: {device}')



model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet50', pretrained=False)
path = '/root/share/origin'

data_transforms = transforms.Compose([
transforms.RandomResizedCrop(224),
transforms.RandomHorizontalFlip(),
transforms.ToTensor(),
transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

image_datasets = datasets.ImageFolder(path,data_transforms)

train_size = int(0.7 * len(image_datasets))
valid_size = int(0.1 * len(image_datasets))
test_size = len(image_datasets) - train_size - valid_size
train_dataset, valid_dataset , test_dataset = torch.utils.data.random_split(image_datasets, [train_size, valid_size, test_size])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64 ,num_workers=6, shuffle = True)
validation_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=64 ,num_workers=4, shuffle = False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, num_workers=6,shuffle=False)

dataloaders={
    'train':train_loader,
    'valid':validation_loader,
    'test':test_loader
}
dataset_sizes={
    'train':train_size,
    'valid':valid_size,
    'test':test_size
}
loss_fn = FocalLoss(device = device, gamma = 2.).to(device)

