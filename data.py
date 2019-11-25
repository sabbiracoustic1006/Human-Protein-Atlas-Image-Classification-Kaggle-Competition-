

from torchvision import transforms
import pandas as pd
import torch, os
from PIL import Image
import torchvision
import torch.nn as nn
from imgaug import augmenters as iaa
import numpy as np
import cv2
from torchvision.transforms import *


lows = 4*[8]+4*[9]+4*[10]+4*[15]+2*[17]+2*[20]+2*[24]+2*[26]+4*[27]


def read_csv(num_classes=28,mode = 'train'):        
    paths = []
    labels = []
    if mode == 'train':
        df = pd.read_csv('/home/ratul/data/human_atlas_data/labels/final1024.csv')[:31060]
        df = df.sort_values(by='Id', inplace=False, ascending=True)
        pathTrain = '/home/ratul/data/torch_train_tensors'
        for name, lbl in zip(df['Id'], df['Target'].str.split(' ')):
            y = np.zeros(num_classes)
            for key in lbl:
                y[int(key)] = 1
            labels.append(y)
            paths.append(os.path.join(pathTrain,name))
    
    return np.array(paths),np.array(labels),df

def dummy_label_create(data):
    paths = []
    labels = []   

    for name, lbl in zip(data['Id'], data['Target'].str.split(' ')):
        y = np.zeros(28)
        for key in lbl:
            y[int(key)] = 1
        paths.append(name)
        labels.append(y)

    return np.array(paths), np.array(labels)




def oversample(data):
        
    data_orig = data.copy() 
    
    for i in lows:
        target = str(i)
        
        indicies = data_orig.loc[data_orig['Target'] == target].index
        data = pd.concat([data,data_orig.loc[indicies]], ignore_index=True)
        
        indicies = data_orig.loc[data_orig['Target'].str.startswith(target+" ")].index
        data = pd.concat([data,data_orig.loc[indicies]], ignore_index=True)
        
        indicies = data_orig.loc[data_orig['Target'].str.endswith(" "+target)].index
        data = pd.concat([data,data_orig.loc[indicies]], ignore_index=True)
        
        indicies = data_orig.loc[data_orig['Target'].str.contains(" "+target+" ")].index
        data = pd.concat([data,data_orig.loc[indicies]], ignore_index=True)
        
    return data



class HumanDataset(torch.utils.data.Dataset):
   
    def __init__(self,id_lists,labels,augment):
        self.id_lists = id_lists
        self.labels = labels
        self.augment = augment
        self.transform = Compose([ToPILImage(),RandomChoice([RandomAffine(degrees=0,shear=(-16,16)),RandomVerticalFlip(),RandomHorizontalFlip()]),ToTensor()])
        
    def __len__(self):
        return len(self.id_lists)
    
    def __getitem__(self,index):
        
        label = self.labels[index]
        label = label.astype('float32')
        
        img = torch.load(self.id_lists[index]).float()/255
        if self.augment:
            img = self.transform(img)
        
        label = torch.tensor(label)
        
        return img,label
    

    def augumentor(self,image):
        augment_img = iaa.Sequential([
            iaa.OneOf([
                iaa.Affine(rotate=90),
                iaa.Affine(rotate=180),
                iaa.Affine(rotate=270),
                iaa.Affine(shear=(-16, 16)),
                iaa.Fliplr(0.5),
                iaa.Flipud(0.5)
                
            ])], random_order=True)
        
        image_aug = augment_img.augment_image(image)
        
        return image_aug

    def labels_binarizer(self,x):
        labels = np.zeros(self.num_classes)
        labels[x] = 1
        
        return labels
    
    def read_image(self,Id):
        npy_path = Id + '.npy'
#        r_chn = Id + '_red.png'
#        g_chn = Id + '_green.png'
#        b_chn = Id + '_blue.png'
#        y_chn = Id + '_yellow.png'

#        r_chn = cv2.imread(r_chn,0)
#        g_chn = cv2.imread(g_chn,0)
#        b_chn = cv2.imread(b_chn,0)
#        y_chn = cv2.imread(y_chn,0)
#
#        r_chn = r_chn[np.newaxis,...]
#        g_chn = g_chn[np.newaxis,...]
#        b_chn = b_chn[np.newaxis,...]
#        y_chn = y_chn[np.newaxis,...]
#        img = np.vstack((r_chn,g_chn))
#        img = np.vstack((img,b_chn))
#        img = np.vstack((img,y_chn))
        img = np.load(npy_path,allow_pickle=True)
        if img.shape[0] == 4:
            img = img.transpose()
        
        return img
    
class HumanDatasetModified(torch.utils.data.Dataset):
   
    def __init__(self,id_lists,labels,augment):
        self.paths = id_lists
        self.labels = labels
        self.augment = augment
        self.transform = transforms.Compose([transforms.RandomAffine(degrees=(-90,90),shear=(-16,16)),transforms.RandomHorizontalFlip(),
                                             transforms.RandomVerticalFlip(),transforms.ToTensor()])
        
        self.to_tensor = transforms.ToTensor()
        
    def __len__(self):
        return len(self.paths)

    def __getitem__(self,idx):
        path = self.paths[idx]
        label = torch.tensor(self.labels[idx])
        extension = ('.png' if 'human_atlas_data' in path else '.jpg')
        pil_image = Image.merge('RGBA',list(map(lambda x:Image.open(path+'_%s%s'%(x,extension)),['red','green','blue','yellow'])))
        if self.augment:
            image = self.transform(pil_image)
        else:
            image = self.to_tensor(pil_image)
        
        return image,label
        
  
        
