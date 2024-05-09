#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 09:40:56 2022

@author: wajihullah.baig
"""

import torch
import pandas as pd
import cv2
import numpy as np
import glob



class GistTrainDataSet(torch.utils.data.Dataset):
    def __init__(self, labels_file, root_dir):
        self.data = pd.read_csv(labels_file)
        self.root_dir = root_dir
        return

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        feature_path = self.root_dir + "/" + self.data['file_path'][idx]
        feature =  torch.tensor(np.load(feature_path), dtype=torch.float32)
        label = self.data['label'][idx]

        return feature, label

class TrainDataSet(torch.utils.data.Dataset):
    def __init__(self,labels_file, root_dir, transforms):
        self.data = pd.read_csv(labels_file)
        self.root_dir = root_dir
        self.transforms = transforms
        return

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image_path = self.root_dir + "/" + self.data['file_path'][idx]
        image = cv2.imread(image_path, 0)
        image = np.float32(image/255.0) # For now lets just normalize everything between 0 and 1
        label = self.data['label'][idx]
        if self.transforms:
            image = self.transforms(image)

        return image, label
    

class TestDataSet(torch.utils.data.Dataset):
    def __init__(self, root_dir, transforms):
        self.data = glob.glob(root_dir)
        self.root_dir = root_dir
        self.transforms = transforms
        return

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image_path = self.data[idx]
        image = cv2.imread(image_path, 0)
        image = np.float32(image/255.0) # For now lets just normalize everything between 0 and 1        
        if self.transforms:
            image = self.transforms(image)

        return image_path,image    
    
    
class GistTestDataSet(torch.utils.data.Dataset):
    def __init__(self, root_dir):
        self.data = glob.glob(root_dir)
        self.root_dir = root_dir
        return

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        feature_path =  self.data[idx]
        feature =  torch.tensor(np.load(feature_path), dtype=torch.float32)
        return feature_path,feature        