#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  1 14:19:50 2019

@author: ratul
"""
import torch, os
import numpy as np 
import pandas as pd 
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from utils import save_checkpoint
from main_mod import evaluate, featExt
from config import config
from torch import nn,optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm_demo import TQDM
from data import HumanDataset,dummy_label_create,oversample,read_csv
from model_resnet18 import model
#%%


def getTrainDataset(df_path):
    df = pd.read_csv(df_path)
    paths = list(df['Id'].values)
    labels = []   
    for lbl in df['Target'].str.split(' '):
        y = np.zeros(28)
        for key in lbl:
            y[int(key)] = 1
        labels.append(y)

    return np.array(paths), np.array(labels)

def getTrainDatasetNpy(df_path):
    df = pd.read_csv(df_path)
    paths = [os.path.join(os.path.abspath('../../data/HPAv18_npy'),path.split('/')[-1]) for path in list(df['Id'].values)]

    labels = []   
    for lbl in df['Target'].str.split(' '):
        y = np.zeros(28)
        for key in lbl:
            y[int(key)] = 1
        labels.append(y)

    return np.array(paths), np.array(labels)



if __name__ == '__main__':
    patience = 5
    earlyThreshold = 0.0005
    num_fold = 5
    mskf = MultilabelStratifiedKFold(n_splits= num_fold, random_state=0)
    
    fold = 0
    X,y,_ = read_csv()

    
    for train_index, val_index in mskf.split(X, y):

    
        X_val, y_val = X[val_index], y[val_index]
        X_train, y_train = X[train_index], y[train_index]
        
        
        print("starting fold: {}".format(fold))

        if not os.path.exists(config.submit):
            os.makedirs(config.submit)
        if not os.path.exists(config.weights + config.model_name + os.sep +str(fold)):
            os.makedirs(config.weights + config.model_name + os.sep +str(fold))
        if not os.path.exists(config.best_models):
            os.mkdir(config.best_models)
        if not os.path.exists("./logs/"):
            os.mkdir("./logs/")
        
        best_loss = 999
        best_f1 = 0
        best_results = [np.inf,0]
        val_metrics = [np.inf,0]
        
        model.load_state_dict(torch.load('checkpoints/best_models/%s_fold_%d_model_best_f1.pth.tar'%(config.model_name,fold))['state_dict'])

        
        model.cuda()
        
        
        criterion = nn.BCEWithLogitsLoss().cuda()
        optimizer = optim.Adam(model.parameters(),lr=1e-3)
        scheduler = ReduceLROnPlateau(optimizer,factor=0.5,patience=2,min_lr=1e-5)
        
        train_gen = HumanDataset(X_train,y_train,augment=True)
        val_gen = HumanDataset(X_val,y_val,augment=False)
        
        train_loader = torch.utils.data.DataLoader(train_gen,batch_size=config.batch_size,shuffle=True,num_workers=6,pin_memory=True)
        val_loader = torch.utils.data.DataLoader(val_gen,batch_size=config.batch_size,num_workers=6,pin_memory=True)
#        
        allPred = featExt(val_loader,model)
        break
        
        loss_history = []
        with TQDM() as pbar:
            pbar.on_train_begin({'num_batches':len(train_loader),'num_epoch':config.epochs})
            for epoch in range(0,config.epochs):        
                training_loss = []
                pbar.on_epoch_begin(epoch)
                print('learning rate is %.5f'%optimizer.state_dict()['param_groups'][0]['lr'])
                model.train()
                for i,(images,target) in enumerate(train_loader):
                        
                    pbar.on_batch_begin()
                    images = images.cuda()
                    
                    target = torch.from_numpy(np.array(target)).float().cuda()
                    output = model(images)
                    loss = criterion(output,target)               
                    optimizer.zero_grad()
                    

                        
                    loss.backward()
                        
                    
                    optimizer.step()
                    training_loss.append(loss.data.cpu().numpy())
                    
                    pbar.on_batch_end(logs={'loss':loss})
                # val
                val_metrics = evaluate(val_loader,model,criterion,epoch,np.average(training_loss),best_results,None)
                pbar.on_epoch_end({'loss':sum(training_loss)/len(training_loss),'val_loss':val_metrics[0],'val_accuracy_.2':val_metrics[1],'val_accuracy_.3':val_metrics[2]})
                scheduler.step(val_metrics[0])
                # check results 
                is_best_loss = val_metrics[0] < best_results[0]
                best_results[0] = min(val_metrics[0],best_results[0])
                is_best_f1 = val_metrics[1] > best_results[1]
                best_results[1] = max(val_metrics[1],best_results[1])  
                
                # save model
                save_checkpoint({
                            "epoch":epoch + 1,
                            "model_name":config.model_name,
                            "state_dict":model.state_dict(),
                            "best_loss":best_results[0],
                            "optimizer":optimizer.state_dict(),
                            "fold":fold,
                            "best_f1":best_results[1],
                },is_best_loss,is_best_f1,fold)
        
                loss_history.append(val_metrics[0])
                if epoch >= patience-1:
                    best = min(loss_history)
                    t = loss_history[epoch-patience+1:epoch+1]
                    differences = [best-i for i in t]
                    count = sum([1 for x in differences if (x >= -earlyThreshold and x<=0)])
                    if count == patience:
                        print('Early stopping')
                        break
         
        torch.save(model.state_dict(),'fold_%d.pth'%fold)
        fold+=1
        if fold==num_fold:
            print("finished training all {} fold".format(num_fold))
            break