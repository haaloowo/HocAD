# -*- coding: utf-8 -*-

import os
import random
import numpy as np
import time
import cv2
import torch
import json

from .data_enhance import rotate_img
from .data_utils import input_transform


def save_score_data(path,dic):
    with open(path,'w') as f:
        f.write(json.dumps(dic))

def random_sel_img_and_read(DATASET_PATH, CATEGORY, SHOT_NUM, INPUT_IMAGE_SIZE, INPUT_EXPAND_ROTATE):
    ###---------------------------random select k-shot image--------------------
    category_train_path = os.path.join(DATASET_PATH,CATEGORY, 'train/good')
    img_name_list = os.listdir(category_train_path)

    random.shuffle(img_name_list)  
    selected_img_name_lst = img_name_list[0:SHOT_NUM]
    print('input images:\n',selected_img_name_lst)
    
    ### ----------------------------load images-------------------------------
    input_img_arr = np.zeros((SHOT_NUM,INPUT_IMAGE_SIZE,INPUT_IMAGE_SIZE,3))
    start_time = time.time()
    for i,img_name in enumerate(selected_img_name_lst):
        # print(img_name)
        img = cv2.imread(os.path.join(category_train_path,img_name))
        img = cv2.resize(img,(INPUT_IMAGE_SIZE,INPUT_IMAGE_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_img_arr[i] = img
    input_img_arr = input_img_arr.astype(np.uint8)
    print('read image time used:',time.time() - start_time)
    
    ### ------------------------normal data enhance-------------------------
    if INPUT_EXPAND_ROTATE:
        for i in range(SHOT_NUM):
            img = input_img_arr[i]
            for rot_angle in [90, 180, 270]:
                img_rot = rotate_img(img, rot_angle)
                img_rot = np.expand_dims(img_rot,axis=0)
                input_img_arr = np.concatenate((input_img_arr, img_rot), axis=0)
    input_img_arr = input_img_arr.astype(np.uint8)
    print('fixed rotate input_img_arr:',input_img_arr.shape)
    
    
    return input_img_arr


def get_feature_extractor_out_size(feature_extractor, INPUT_IMAGE_SIZE, device):
    with torch.no_grad():
        x = torch.rand(1, 3, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE).to(device)
        feature = feature_extractor.get_feature(x)
        _,C,H,W = feature.shape
        
    return C,H,W


def parse_gt_mask(gt,patch_num = 14,patch_size=16):
    patch_mask_arr = np.zeros(patch_num ** 2)
    for i in range(patch_num):
        for j in range(patch_num):
            gt_patch = gt[patch_size*j:patch_size*(j+1), patch_size*i:patch_size*(i+1)]
            n = np.sum(gt_patch)
            patch_mask_arr[patch_num*j+i] = n
    return patch_mask_arr.astype(int)



no_foreground_category_lst = ['leather', 'tile', 'carpet', 'wood', 'grid', 'zipper','bottle']
read_saved_foreground_category_lst = ['cable', 'pill', 'toothbrush', 'transistor', 'capsule']
inverse_foreground_category_lst = ['hazelnut', 'metal_nut']


def foreground_mask_init(category, img_size=(224,224)):
    target_foreground_mask = cv2.imread('./foreground_mask/{}.bmp'.format(category),
                                        cv2.IMREAD_GRAYSCALE)
    target_foreground_mask = cv2.resize(target_foreground_mask, img_size)
    rd_target_foreground_mask = target_foreground_mask.astype(bool).astype(int)
    return rd_target_foreground_mask


def get_foreground_patch_idx(category, foreground_patch_thres=0):
    target_foreground_mask = foreground_mask_init(category)
    patch_counter_arr = parse_gt_mask(target_foreground_mask)
    idx_arr = np.where(patch_counter_arr > foreground_patch_thres)[0]
    return idx_arr

def gen_normal_feat_bank(category, 
                         input_img_arr, 
                         feature_extractor,
                          H, W, C, 
                          device):
    avg_pool_func = torch.nn.AvgPool2d(3, 1, 1)  

    if category in read_saved_foreground_category_lst:
        USE_FOREGROUND = True
        foreground_patch_idx = get_foreground_patch_idx(category)
        print('foreground_patch_idx:',foreground_patch_idx.shape[0])
        foreground_patch_num = foreground_patch_idx.shape[0]
    else:
        USE_FOREGROUND = False
        foreground_patch_idx = None
    
    with torch.no_grad():
        input_img_num = input_img_arr.shape[0]  
        
        normal_feature_bank = torch.zeros((input_img_num, H*W, C))
        if USE_FOREGROUND:
            foreground_normal_feature_bank = torch.zeros((input_img_num, foreground_patch_num, C))
        else:
            foreground_normal_feature_bank = None
            
        for i in range(input_img_num):
            x = input_transform(input_img_arr[i]).float() 
            x = x.unsqueeze(0).to(device)
            
            feature = feature_extractor.get_feature(x) 
            
            avg_pool_feat = avg_pool_func(feature)
            if i == 0:
                normal_avg_pool_feat_bank = avg_pool_feat
            else:
                normal_avg_pool_feat_bank = torch.cat([normal_avg_pool_feat_bank, avg_pool_feat], dim=0)
            
            feature = feature.cpu().squeeze().reshape(C,-1).permute(1,0)  
            
            normal_feature_bank[i] = feature
            if USE_FOREGROUND:
                foreground_normal_feature_bank[i] = feature[foreground_patch_idx]
     
        normal_avg_pool_feat_bank = normal_avg_pool_feat_bank.cpu().reshape(input_img_num,C,-1)
        normal_avg_pool_feat_bank = normal_avg_pool_feat_bank.permute(0,2,1).reshape(-1,C)
        
        normal_feature_bank = normal_feature_bank.reshape(-1,C)
        
        if USE_FOREGROUND:
            foreground_normal_feature_bank = foreground_normal_feature_bank.reshape(-1,C)
    
    return normal_feature_bank, foreground_normal_feature_bank, foreground_patch_idx
    



