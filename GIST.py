#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 08:51:02 2022

@author: wajihullah.baig
"""

# https://github.com/ufoym/pyGIST/edit/master/gist.py

import numpy as np
import matplotlib.pyplot as plt
import cv2

# ----------------------------------------------------------------------------


def gist(img, image_size=128, orientations=(8, 8, 4), num_blocks=4):

    # ------------------------------------------------------------------------

    def prefilter(img, fc=4, w=5):
        s1 = fc / np.sqrt(np.log(2))

        # Pad images to reduce boundary artifacts
        img = np.log(img+1)
        img = np.lib.pad(img, ((w, w), (w, w)), 'symmetric')
        sn, sm = img.shape
        n = max((sn, sm))
        n = n + n % 2
        img = np.lib.pad(img, ((0, n-sn), (0, n-sm)), 'symmetric')

        # Filter
        fx, fy = np.meshgrid(np.arange(-n/2, n/2), np.arange(-n/2, n/2))
        gf = np.fft.fftshift(np.exp(-(fx ** 2 + fy ** 2) / (s1 ** 2)))

        # Whitening
        output = img - np.real(np.fft.ifft2(np.multiply(np.fft.fft2(img), gf)))

        # Local contrast normalization
        tmp = np.fft.ifft2(np.multiply(np.fft.fft2(output ** 2), gf))
        localstd = np.sqrt(np.abs(tmp))
        output /= (0.2 + localstd)

        # Crop output to have same size than the input
        output = output[w:sn-w, w:sm-w]
        return output

    # ------------------------------------------------------------------------

    def get_feature(img, w, G):
        '''Estimate global features.'''

        def average(x, N):
            '''Average over non-overlapping square image blocks.'''
            nx = np.fix(np.linspace(0, x.shape[0], N+1)).astype(np.int)
            ny = np.fix(np.linspace(0, x.shape[1], N+1)).astype(np.int)
            y = np.zeros((N, N))
            for xx in range(N):
                for yy in range(N):
                    v = np.mean(x[nx[xx]:nx[xx+1], ny[yy]:ny[yy+1]])
                    y[yy, xx] = v
            return y

        n, n, num_filters = G.shape
        W = w * w
        g = np.zeros((W * num_filters, 1))

        img = np.fft.fft2(img)
        k = 0
        for n in range(num_filters):
            ig = np.abs(np.fft.ifft2(np.multiply(img, G[:, :, n])))
            v = average(ig, w)
            g[k:k+W, :] = np.reshape(v, (W, 1))
            k += W
        return g

    # ------------------------------------------------------------------------

    def create_gabor(orientations, n):
        '''Compute filter transfer functions.'''

        Nscales = len(orientations)
        num_filters = sum(orientations)

        param = []
        for i in range(Nscales):
            for j in range(orientations[i]):
                param.append([.35,
                              .3/(1.85 ** i),
                              16.0 * orientations[i] ** 2 / 32 ** 2,
                              np.pi / orientations[i] * j])
        param = np.array(param)

        # Frequencies:
        fx, fy = np.meshgrid(np.arange(-n/2, n/2), np.arange(-n/2, n/2))
        fr = np.fft.fftshift(np.sqrt(fx ** 2 + fy ** 2))
        t = np.fft.fftshift(np.angle(fx + 1j * fy))

        # Transfer functions:
        gabors = np.zeros((n, n, num_filters))
        for i in range(num_filters):
            tr = t + param[i, 3]
            tr = tr + 2 * np.pi * (tr < -np.pi) - 2 * np.pi * (tr > np.pi)
            gabors[:, :, i] = np.exp(
                - 10 * param[i, 0] * (fr / n / param[i, 1] - 1) ** 2
                - 2 * param[i, 2] * np.pi * tr ** 2)
        return gabors

    # ------------------------------------------------------------------------

    img = cv2.resize(img, (image_size, image_size))
    gabors = create_gabor(orientations, image_size)
    output = prefilter(img.astype(np.float))
    features = get_feature(output, num_blocks, gabors)
    return features.flatten().squeeze()


# ----------------------------------------------------------------------------


if __name__ == '__main__':
    import pandas as pd
    import time
    from pathlib import Path
    import os
    root = "dataset"
    labels_file = "train_augumented2_labels.csv"
    labels_df = pd.read_csv(root+"/"+labels_file)
    gist_labels_df = labels_df.copy()
    output_dir = "gist_features/"
    Path(root+"/"+output_dir).mkdir(parents=True, exist_ok=True)
    for index,row in labels_df.iterrows():
        print("Processing:",row["file_path"])
        img = cv2.imread(root + "/" +row["file_path"],0)
        ts = time.time()
        gist_feature =  gist(img,image_size=28)
        file_path = root + "/"+output_dir+os.path.basename(row["file_path"])
        np.save(file_path,gist_feature)
        gist_labels_df.at[index,"file_path"]  = output_dir+os.path.basename(row["file_path"]+".npy"  )    
        #print(gabor_feature)
        #fig, ax = plt.subplots()
        #im = ax.plot(gabor_feature)
        #plt.show()
         
        print (time.time() - ts)
    gist_labels_df.to_csv("dataset/train_gist_augumented2_labels.csv",index=False)
# ----------------------------------------------------------------------------
