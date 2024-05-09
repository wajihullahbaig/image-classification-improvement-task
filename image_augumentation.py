#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 09:40:56 2022

@author: wajihullah.baig
"""


import cv2
import numpy as np
from skimage.util import random_noise


def flip(image, vflip=False, hflip=False):
        '''
        Flip the image
        :param image: image to be processed
        :param vflip: whether to flip the image vertically
        :param hflip: whether to flip the image horizontally
        '''
        if hflip or vflip:
            if hflip and vflip:
                c = -1
            else:
                c = 0 if vflip else 1
            image = cv2.flip(image, flipCode=c)
        return image 
    
def rotate(image, angle=90, scale=1.0):
        '''
        Rotate the image
        :param image: image to be processed
        :param angle: Rotation angle in degrees. Positive values mean counter-clockwise rotation (the coordinate origin is assumed to be the top-left corner).
        :param scale: Isotropic scale factor.
        '''
        
        w = image.shape[1]
        h = image.shape[0]
        #rotate matrix
        M = cv2.getRotationMatrix2D((w/2,h/2), angle, scale)
        #rotate
        image = cv2.warpAffine(image,M,(w,h))
        return image

def gausian_blur(image,blur):
    image = cv2.GaussianBlur(image,(5,5),blur)
    return image;

def averageing_blur(image,kenel_size):
    image=cv2.blur(image,(kenel_size,kenel_size))
    return image

def median_blur(image,kenel_size):
    image=cv2.medianBlur(image,kenel_size)
    return image

def bileteralBlur(image,d,color,space):
    image = cv2.bilateralFilter(image, d,color,space)
    return image

def erosion_image(image,kernel_size):
    kernel = np.ones((kernel_size,kernel_size),np.uint8)
    image = cv2.erode(image,kernel,iterations = 1)
    return image
def dilation_image(image,kenel_size):
    kernel = np.ones((kenel_size, kenel_size), np.uint8)
    image = cv2.dilate(image,kernel,iterations = 1)
    return image

def translation_image(image,x,y):
    rows, cols = image.shape
    M = np.float32([[1, 0, x], [0, 1, y]])
    image = cv2.warpAffine(image, M, (cols, rows))
    return image

def salt_and_pepper_noise(image, amount):
    # Add salt-and-pepper noise to the image.
    noise_img = random_noise(image, mode='s&p',amount=amount)
    
    # The above function returns a floating-point image
    # on the range [0, 1], thus we changed it to 'uint8'
    # and from [0,255]
    noise_img = np.array(255*noise_img, dtype = 'uint8')
    return noise_img
