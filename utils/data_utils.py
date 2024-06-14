# -*- coding: utf-8 -*-
import torch
import torchvision.transforms as transforms
import numpy as np
import random
import matplotlib.pyplot as plt
import os
import time

input_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean = (0.485, 0.456, 0.406),
        std  = (0.229, 0.224, 0.225))
    ])




def gen_abnormal_feat_bank(abnormal_feat_num,
                           feat_dim,
                           input_img_arr,
                           PATCH_ABNORMAL_THRESHOLD,
                           PATCH_ABNORMAL_THRESHOLD_MAX,
                           feature_extractor,
                           anormaly_generator,
                           abnormal_feat_save_path,
                           device,
                           SAVE_ABNORMAL_FEAT = True,
                           USE_DIFF_FEATURE = False,
                           patch_avg_proto_feat = None,
                           ):
    start_time = time.time()
    if os.path.exists(abnormal_feat_save_path):
        print('Read exists abnormal feature bank....')
        abnormal_feature_bank = np.load(abnormal_feat_save_path)
        abnormal_feature_bank = torch.from_numpy(abnormal_feature_bank)
    else:
        print('Gen abnormal feature bank....')
        if USE_DIFF_FEATURE:
            abnormal_feature_bank = torch.zeros((abnormal_feat_num, feat_dim * 2))
        else:
            abnormal_feature_bank = torch.zeros((abnormal_feat_num, feat_dim))

        counter = 0 ### abnormal feature num
        N = input_img_arr.shape[0]  
        img_idx = 0
        with torch.no_grad():
            while True:
                img = input_img_arr[img_idx % N]
                img, mask = anormaly_generator(img=img)
                
                patch_mask = parse_gt_mask(mask, patch_num = 14, patch_size=16)
                idx_arr = np.where(patch_mask > PATCH_ABNORMAL_THRESHOLD)[0]
                idx_arr_ = np.where(patch_mask < PATCH_ABNORMAL_THRESHOLD_MAX)[0]

                patch_idx_lst = [val for val in idx_arr if val in idx_arr_]
                idx_arr = np.array(patch_idx_lst)

                if idx_arr.shape[0] > 0:
                    x = input_transform(img)
                    x = x.float() 
                    x = x.unsqueeze(0).to(device)
                    feature = feature_extractor.get_feature(x)  

                    feature = feature.cpu().squeeze().reshape(feat_dim, -1).permute(1, 0) 
                    diff_feature = torch.abs(feature - patch_avg_proto_feat)
                    if USE_DIFF_FEATURE:
                        feature = torch.cat([feature, diff_feature], dim=-1)
                        
                    for idx in idx_arr:
                        abnormal_feature_bank[counter] = feature[idx]
                        counter += 1
                        
                        if counter == abnormal_feat_num:
                            break
                if counter == abnormal_feat_num:
                    break
                img_idx += 1
                
        if SAVE_ABNORMAL_FEAT:
            np.save(abnormal_feat_save_path, abnormal_feature_bank.numpy())
        print('gen anormaly bank time used:',time.time() - start_time)
    return abnormal_feature_bank



def parse_gt_mask(gt,patch_num = 14,patch_size=16):
    patch_mask_arr = np.zeros(patch_num ** 2)
    for i in range(patch_num):
        for j in range(patch_num):
            gt_patch = gt[patch_size*j:patch_size*(j+1), patch_size*i:patch_size*(i+1)]
            n = np.sum(gt_patch)
            patch_mask_arr[patch_num*j+i] = n
    return patch_mask_arr.astype(int)


def get_feat_set(normal_feature_bank, 
                    abnormal_feature_bank, 
                    SUPPORT_SET_NUM,
                    random_out = True
                    ):
    
    normal_feat_num = normal_feature_bank.shape[0]
    abnormal_feat_num = abnormal_feature_bank.shape[0]
    
    normal_sel_num = int(SUPPORT_SET_NUM*0.5)
    abnormal_sel_num = SUPPORT_SET_NUM - normal_sel_num
    
    normal_sel_index = torch.LongTensor(random.sample(range(normal_feat_num), normal_sel_num))
    normal_support_feat = torch.index_select(normal_feature_bank, 0, normal_sel_index)

    abnormal_sel_index = torch.LongTensor(random.sample(range(abnormal_feat_num), abnormal_sel_num))
    abnormal_support_feat = torch.index_select(abnormal_feature_bank, 0, abnormal_sel_index)
    
    
    support_feat = torch.cat((normal_support_feat, abnormal_support_feat), dim=0)
    support_label = torch.cat((torch.zeros(normal_sel_num), 
                               torch.ones(abnormal_sel_num)), dim=0)
    if random_out: 
        rand_idx = torch.randperm(SUPPORT_SET_NUM)
        support_feat = support_feat[rand_idx]
        support_label = support_label[rand_idx]
    return support_feat, support_label, normal_sel_index


def get_test_feat_set(
                normal_feature_bank, 
                abnormal_feature_bank, 
                TEST_SUPPORT_SET_NUM
        ):
    test_support_set_normal_feat_num = int(TEST_SUPPORT_SET_NUM * 0.5)
    test_support_set_abnormal_feat_num = int(TEST_SUPPORT_SET_NUM * 0.5)
    

    normal_feat_selected_idx = None
    normal_support_feat = normal_feature_bank[0:test_support_set_normal_feat_num]


    
    abnormal_feat_num = abnormal_feature_bank.shape[0]
    index = torch.LongTensor(random.sample(range(abnormal_feat_num), test_support_set_abnormal_feat_num))
    abnormal_support_feat = torch.index_select(abnormal_feature_bank, 0, index)
    
    support_feat = torch.cat((normal_support_feat, abnormal_support_feat), dim=0)
    return support_feat, normal_feat_selected_idx


