from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from skimage import io
import torchvision.transforms.functional as TF
from PIL import Image 
import random 

import warnings
warnings.filterwarnings("ignore")

class SimulationsDataset(Dataset):

    def __init__(self, image_file, target_file, train, batch):
        with open(image_file) as file:
            lines = [line.split()[0] for line in file]
        clust = []
        obs = [] 
        for i in range(len(lines)):
            clust.append(lines[i].split('/',1)[0])
            obs.append(lines[i].split('/',1)[1])
        self.clust = clust 
        self.obs = obs 
        self.target_file = target_file 
        self.train = train 
        self.batch = batch 

    def transform(self, image): 
        image = TF.center_crop(image, 60)
        image = TF.to_tensor(image)
        image = TF.normalize(image, [0.5], [0.5])

        return image 
        
    def __getitem__(self,index):
        if (self.train == 0):  
            index = (index + self.batch*50)%250
            im_name = 'images/smoothed_img' + self.clust[index] + self.obs[index] + '.png'
        elif (self.train==1):
            index = (index + self.batch*50 + 200)%250 
            im_name = 'images/smoothed_img' + self.clust[index] + self.obs[index] + '.png'
        else:
            index = 250+index
            im_name = 'images/smoothed_img' + self.clust[index] + self.obs[index] + '.png'
        image = Image.open(im_name)
        image = image.convert('L')
        image = image.resize((70,70))
        image = self.transform(image)

        labels = np.loadtxt(self.target_file) 
        label = (labels[index]) 
        return image, label  
    
    def __len__(self): 
        if (self.train==0): 
            size = 200 
        elif (self.train==1): 
            size = 50
        else:
            size = 58       
        return size 


