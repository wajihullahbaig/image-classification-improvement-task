#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  1 11:15:34 2022

@author: wajihullah.baig
"""

import pandas as pd
from pathlib import Path
import os
import image_augumentation as ia
import cv2

root = "dataset"

input_dataset_path = root+"/train_augumented/"
output_dataset_path = root +"/train_augumented2/"
labels_file = root+"/train_augumented_labels.csv"

labels_df= pd.read_csv(labels_file);
Path(output_dataset_path).mkdir(parents=True, exist_ok=True)



labels_augumented = labels_df.copy()
for index,row in labels_df.iterrows():
    print("Processing item:",row["file_path"])
    img =  cv2.imread(root+"/"+row["file_path"],0)
    file_name = "dilated_"+os.path.basename(row["file_path"])
    img2 = ia.dilation_image(img, 1)
    cv2.imwrite(output_dataset_path+file_name,img2)         
    labels_augumented.loc[-1] = ["train_augumented2/"+file_name,row["label"]]
    labels_augumented.index = labels_augumented.index + 1  # shifting index
    
    img =  cv2.imread(root+"/"+row["file_path"],0)
    file_name = "eroded_"+os.path.basename(row["file_path"])
    img2 = ia.erosion_image(img, 1)
    cv2.imwrite(output_dataset_path+file_name,img2)         
    labels_augumented.loc[-1] = ["train_augumented2/"+file_name,row["label"]]
    labels_augumented.index = labels_augumented.index + 1  # shifting index
    
    file_name = "translated_"+os.path.basename(row["file_path"])
    img2 = ia.translation_image(img, 2,2)
    cv2.imwrite(output_dataset_path+file_name,img2)         
    labels_augumented.loc[-1] = ["train_augumented2/"+file_name,row["label"]]
    labels_augumented.index = labels_augumented.index + 1  # shifting index
    
    
    file_name = "gausian_blur_"+os.path.basename(row["file_path"])
    img2 = ia.gausian_blur(img, 1)
    cv2.imwrite(output_dataset_path+file_name,img2)         
    labels_augumented.loc[-1] = ["train_augumented2/"+file_name,row["label"]]
    labels_augumented.index = labels_augumented.index + 1  # shifting index
    
    file_name = "salt_and_pepper_noise_"+os.path.basename(row["file_path"])
    img2 = ia.salt_and_pepper_noise(img,0.05)
    cv2.imwrite(output_dataset_path+file_name,img2)         
    labels_augumented.loc[-1] = ["train_augumented2/"+file_name,row["label"]]
    labels_augumented.index = labels_augumented.index + 1  # shifting index
    
        
    labels_augumented = labels_augumented.sort_index()

labels_augumented['file_path'] = labels_augumented['file_path'].str.replace('train_augumented/','train_augumented2/')
labels_augumented.to_csv("dataset/train_augumented2_labels.csv",index=False)
# Lets copy the original dataset to the balanced dataset folder to even things completely
import shutil
shutil.copytree(input_dataset_path, output_dataset_path, dirs_exist_ok=True)