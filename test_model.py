#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 00:11:31 2018

@author: ratul
"""
import pandas as pd
from sklearn.metrics import f1_score
import scipy.optimize as opt
import torch, os
import numpy as np
from tqdm import tqdm
from functions import getTestDataset, load_test_image, Classifier, getTrainDataset, HumanAtlasDataset, ToTensor
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from models.model import*
from model_resnet18 import model

cores = 6
batch_size = 24
os.makedirs('submissions',exist_ok=True)
cuda = False


classifier = model
if cuda:
    classifier.cuda()
    
fold = 0
best_model = torch.load("%s/%s_fold_%s_model_best_loss.pth.tar"%(config.best_models,config.model_name,str(fold)))
classifier.load_state_dict(best_model["state_dict"])

    
#%% FInd best Threshold value
    
paths,labels = getTrainDataset()
labels_new = np.sum(labels,axis=1)


train_paths,validation_paths,train_labels,validation_labels = train_test_split(paths,labels,test_size=0.07,random_state=4,stratify=labels_new)

validation_dataset = HumanAtlasDataset(paths=validation_paths, labels=validation_labels,
                                        transform=transforms.Compose([ToTensor()]))

validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size,
                            shuffle=True, num_workers=cores)
final_f1 = []
for t in np.arange(0.1,0.9,0.05):
    f1_scores = []
    for sample in tqdm(validation_dataloader,total=len(validation_dataloader)):
        with torch.no_grad():
            im,label = sample['image'].float().cuda(),sample['label']
            predicted = classifier(im).sigmoid().data.cpu().numpy()>t
            f1 = f1_score(predicted,label.numpy(),average='macro')
            f1_scores.append(f1)
    print('f1 score for threshold %.1f is %.3f'%(t,sum(f1_scores)/len(f1_scores)))
    final_f1.append(sum(f1_scores)/len(f1_scores))


#%% Test section

th_t = 0.11

paths = getTestDataset()
labelizer = lambda x: ' '.join(list(map(str,x[0])))


for param in classifier.parameters():
    param.requires_grad = False
    
test_dataset = HumanAtlasDataset(paths=paths,
                                 transform=transforms.Compose([ToTensor()]),
                                 test=True)

test_dataloader = DataLoader(test_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=cores)

  
final_labels = []
for sample in tqdm(test_dataloader,total=len(test_dataloader)):
    with torch.no_grad():
        im = sample['image']
        predicted = classifier(im.float().cuda()).sigmoid()
        for p in predicted:
            labels = np.nonzero((p.data.cpu().numpy()>th_t).reshape(-1))
            final_labels.append(labelizer(labels))
        
            
dic = {'Id':list(map(lambda x: x.replace('/home/ratul/data/human_atlas_data/test/',''),paths)),'Predicted':final_labels}
pd.DataFrame(dic).to_csv('submissions/bninception_%.2f.csv'%th_t,index=False)