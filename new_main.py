import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models 
from torchvision.datasets import ImageFolder
import torch.utils.data as data
from torchvision import datasets, transforms
import numpy as np
import pandas as pd
from losses import FocalLoss
from torch.optim import lr_scheduler
from new_train import train_model, test_model

print(torch.cuda.is_available())
device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")
print(f'\ndevice: {device}')

parameter='train'

model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet50', pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 15)
ckpt = torch.load('/root/share/result/resnet50_models/stage1_1e-05_50.pth')
model.load_state_dict(ckpt['model'].state_dict())
path = '/root/share/origin'
lr=1e-5

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
#loss_fn = FocalLoss(device = device, gamma = 2.).to(device)
loss_fn = nn.CrossEntropyLoss()
num_ftrs = model.fc.in_features
# 여기서 각 출력 샘플의 크기는 2로 설정합니다.
# 또는, nn.Linear(num_ftrs, len (class_names))로 일반화할 수 있습니다.
# normal, abanormal 의 2개 라벨로 분류하므로 최종 output을 2로 튜닝
model.fc = nn.Linear(num_ftrs, 2)

model = model.to(device)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = lr)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

print("let's start!")

if parameter=='train':
    model = train_model(model, device, loss_fn, optimizer, exp_lr_scheduler, dataloaders, dataset_sizes,
                       num_epochs=50)
else:
    test_model(model, dataloaders)


#학습 및 테스트까지 1차 구현
#에폭 한번 돌때마다 체크포인트 세이브 하는방법 찾아서 구현할것
