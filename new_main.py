import torch
import torch.nn as nn
import torchvision.models as models 
from torchvision.datasets import ImageFolder
import torch.utils.data as data
from torchvision import transforms

import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2

model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet50', pretrained=False)

