import torch
import torchvision
from torch import nn

path = "checkpoints/model_best.pth.tar"

model = torchvision.models.resnet18(pretrained=True)
model.conv1 = nn.Conv2d(4,64,kernel_size=(7, 7),stride=(2, 2),padding=(3, 3),bias=False)
model.avgpool = nn.AdaptiveAvgPool2d(1)
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(nn.BatchNorm1d(num_ftrs,num_ftrs),nn.Dropout(0.25),nn.Linear(num_ftrs, 28))

