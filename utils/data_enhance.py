# -*- coding: utf-8 -*-

import cv2
import numpy as np
import torch


def rotate_img(img, rot):
    if rot == 0: 
        return img
    elif rot == 90: 
        return np.flipud(np.transpose(img, (1,0,2)))
    elif rot == 180: 
        return np.fliplr(np.flipud(img))
    elif rot == 270: 
        return np.transpose(np.flipud(img), (1,0,2))
    


    

        
        
        
        
        
        
        
        
        
        