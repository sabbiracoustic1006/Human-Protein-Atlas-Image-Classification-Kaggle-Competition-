#from torchvision import models
##from pretrainedmodels.models import densenet201
#from torchvision import models
#from torch import nn
##from config import config
#from collections import OrderedDict
#import torch.nn.functional as F
#
#
#class Densenet_modified(nn.Module):
#    def __init__(self):
#        super(Densenet_modified,self).__init__()
#        self.features = models.densenet201(pretrained=True).features
#        self.features.conv0 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
#        self.avg = nn.AdaptiveAvgPool2d(1)
#        self.fc =  nn.Sequential(
#                nn.Dropout(0.5),
#                nn.Linear(1920, 28))
#        
#    def forward(self,img):
#        out = self.avg(self.features(img)).view(-1,1920)
#        out = self.fc(out)
#        return out
#
#def get_net():
#    model = Densenet_modified()
##    model.features.conv0 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
##    model.conv1_7x7_s2 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
##    model.last_linear = nn.Sequential(
##                nn.AdaptiveAvgPool2d(1),
##                nn.BatchNorm1d(1920),
##                nn.Dropout(0.5),
##                nn.Linear(1920, 28),
##            )
#    return model
from torchvision import models
from pretrainedmodels.models import bninception
from pretrainedmodels.models import resnet18
from torch import nn
#from config import config
from collections import OrderedDict
import torch.nn.functional as F

def get_net():
    model = bninception(pretrained="imagenet")
    model.global_pool = nn.AdaptiveAvgPool2d(1)
    model.conv1_7x7_s2 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
    model.last_linear = nn.Sequential(
                nn.BatchNorm1d(1024),
                nn.Dropout(0.5),
                nn.Linear(1024, 28),
            )
    return model

def resnet18_modified():
    resNet = resnet18(pretrained='imagenet')
    resNet.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
    resNet.avgpool = nn.AdaptiveAvgPool2d(1)
    resNet.last_linear = nn.Sequential(
                nn.BatchNorm1d(512),
                nn.Dropout(0.25),
                nn.Linear(512, 28),
            )
    return resNet
    
class simpleModel(nn.Module):
    def __init__(self):
        super(simpleModel,self).__init__()
        self.features = nn.Sequential(nn.Conv2d(4, 16, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2)),nn.ReLU(),nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1),
                                      nn.Conv2d(16, 32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),nn.ReLU(),nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1),
                                      nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),nn.ReLU(),nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                      nn.ReLU(),nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),nn.ReLU(),
                                      nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1))
                                      
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Dropout(0.1),nn.Linear(128,28))
        
        
    def forward(self,x):
        out = self.avg_pool(self.features(x)).view(-1,128)
        out = self.classifier(out)
        return out
    
    
#class ResGAPnet(nn.Module):
#    def __init__(self):
#        super(ResGAPnet,self).__init__()
        