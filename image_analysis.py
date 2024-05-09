#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 07:03:15 2022

@author: wajihullah.baig
"""
import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


ds_path = "dataset/"
train_ds_path = ds_path + "train"
test_ds_path = ds_path + "test"
labels_file = ds_path+"train_labels.csv"



labels_df= pd.read_csv(labels_file);


# Lets see if we have a equal distribution of the classes
# We find the total unique items and plot a histogram
counts = labels_df["label"].value_counts()
labels_df["label"].plot.hist(bins=counts.size, alpha=0.5, stacked=True, width=1)

diffs = counts.max()-counts
print("Class  differences:")
print(diffs)
pers =counts/counts.sum()
print("Class  percentages:")
print(pers)


# It is evident from the histogram and the percentages that we have class imbalanece. 
# Now there are two ways forward, used F1 Score metric in the classifier since
# we have class imbalances, but we need to use Accuracy as a measure.
# Therefore we have to balanace the class distributions. Otherwise out classifier 
# will be biased towards classes with higher samples

# For class distributions we can use, upsampling. The reaseson being the samples for each class are very 
# low and we donot want to use less samples when we perform training.

# Another better option to use is to augument the data before training and balanace the classes.

# But first lets visualize the classes by taking a sample

#sample_path= labels_df[labels_df['label']==0].sample(n=1)["file_path"].values[0]
#data = plt.imread(ds_path+sample_path)
#fig, ax = plt.subplots()
#im = ax.imshow(data, interpolation='none',cmap = 'gray')
#plt.show()
 

# Lets do some simple augumenation and balanace the classes
import image_augumentation as ia
from pathlib import Path
import os
auguemnted_ds_path = "dataset/train_augumented/"
Path(auguemnted_ds_path).mkdir(parents=True, exist_ok=True)
for items in diffs.iteritems():
    if items[1] > 0: 
        samples= labels_df[labels_df['label']==items[0]].sample(n=items[1])
        for index, sample in samples.iterrows():
            data = plt.imread(ds_path+sample["file_path"])
            data = ia.flip(data,hflip=False)
            #fig, ax = plt.subplots()
            #im = ax.imshow(data, interpolation='none',cmap = 'gray')
            #plt.show()
            # save augumented image
            output_path = auguemnted_ds_path+"aug_"+os.path.basename(sample["file_path"])
            plt.imsave(output_path,data)            
            # update the labels 
            labels_df.loc[-1] = ["train_augumented/"+"aug_"+os.path.basename(sample["file_path"]),sample["label"]]
            labels_df.index = labels_df.index + 1  # shifting index
            labels_df = labels_df.sort_index() 

labels_df['file_path'] = labels_df['file_path'].str.replace('train/','train_augumented/')
labels_df.to_csv("dataset/train_augumented_labels.csv",index=False)

# Lets check again for class distribution
# We find the total unique items and plot a histogram
counts = labels_df["label"].value_counts()
fig, ax = plt.subplots()
labels_df["label"].plot.hist(bins=counts.size, alpha=0.5, stacked=True, width=1)
plt.show()

diffs = counts.max()-counts
print("Class  differences:")
print(diffs)
pers =counts/counts.sum()
print("Class  percentages:")
print(pers)

# Lets copy the original dataset to the balanced dataset folder to even things completely
import shutil
shutil.copytree(train_ds_path, train_ds_path, dirs_exist_ok=True)
