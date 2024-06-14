# -*- coding: utf-8 -*-
import os
import time
import numpy as np
import cv2
import json
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from sklearn.metrics import roc_auc_score
from scipy.ndimage import gaussian_filter
from sklearn.metrics import auc
from statistics import mean
from skimage import measure
import pandas as pd
from numpy import ndarray
from utils import input_transform


USE_PATCH_MIN_DIST = True
L2_DIST_WEIGHT = 0.3


inv_normalize = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255], std=[1/0.229, 1/0.224, 1/0.255])

def cal_l2_dist(query, memory, feat_H, feat_W):
    query_feat_num = query.shape[0]
    mem_feat_num = memory.shape[0]

    l2_dist_score = torch.zeros((query_feat_num, mem_feat_num))
    for i in range(query_feat_num):
        diff = F.mse_loss(
            input = torch.repeat_interleave(query[i].unsqueeze(0), repeats=mem_feat_num, dim=0), 
            target = memory,
            reduction ='none'
        )
        diff = diff.sum(dim=1)
        l2_dist_score[i] = diff
    l2_dist_score = torch.min(l2_dist_score, dim=1)[0]  
    return l2_dist_score

def get_patch_proto(memory_feat, feat_H, feat_W):
    feat_num, feat_dim = memory_feat.shape
    img_num = int(feat_num / (feat_H * feat_W))
    memory_feat = memory_feat.reshape(img_num, feat_H * feat_W, feat_dim)
    proto_feat = memory_feat.mean(dim=0)
    return proto_feat

def cal_patch_proto_dist(query, proto):
    diff = query - proto
    patch_l2_dist = diff.norm(p=2, dim=-1)
    return patch_l2_dist


def cvt2heatmap(gray):
    heatmap = cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)  
    return heatmap

def heatmap_on_image(heatmap, image):
    if heatmap.shape != image.shape:
        heatmap = cv2.resize(heatmap, (image.shape[0], image.shape[1]))
    out = np.float32(heatmap)/255 + np.float32(image)/255
    out = out / np.max(out)
    return np.uint8(255 * out)

def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    return (image-a_min)/(a_max - a_min)    


def evaluate(
        feature_extractor,
        corre_module,
        feat_dim,
        feat_H,
        feat_W,
        test_data_loader,
        INPUT_IMAGE_SIZE,
        CATEGORY,
        TEST_SUPPORT_SET_NUM,
        test_support_feat,
        device,
        patch_avg_proto
        ):
    
    print('\n-----evaluate------')
    patch_avg_proto = patch_avg_proto.to(device).unsqueeze(dim=0)
    eps = 1e-6 
    
    corre_module.eval()
    img_save_base_path = './log/output/{}'.format(CATEGORY)
    if not os.path.exists(img_save_base_path):
        os.mkdir(img_save_base_path)

    support_feat = test_support_feat.unsqueeze(dim=0).to(device)
    print('support_feat:',support_feat.shape)

    img_path_lst, gt_path_lst, label_lst, defect_type_lst = test_data_loader.load_dataset()

    aupro_list = []
    gt_list_px_lvl = []
    pred_list_px_lvl = []
    
    gt_list_img_lvl = []
    pred_list_img_lvl = []
    
    
    data_out_dic = {}
    
    with torch.no_grad():
        for i,img_path in enumerate(img_path_lst):
            img = cv2.imread(img_path)
            img = cv2.resize(img,(INPUT_IMAGE_SIZE,INPUT_IMAGE_SIZE))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
            x = input_transform(img).float() 
            x = x.unsqueeze(0).to(device)
            feature = feature_extractor.get_feature(x) 

            test_qry_feature = feature.reshape(1,feat_dim,-1).permute(0,2,1) 
            
           
            diff_feat = torch.abs(test_qry_feature - patch_avg_proto)
            
            test_qry_feature = torch.cat([test_qry_feature, diff_feat], dim=-1)
                    
            
            pred_out_lst = corre_module(test_qry_feature, 
                                        support_feat
                                        ) 
            
            
            pred = torch.cat([torch.sigmoid(pred.unsqueeze(dim=-1)) for pred in pred_out_lst], dim=-1)  ## torch.Size([1, 196, 256, 5])
            pred = torch.sum(pred,dim=-1)

            patch_normal_score = pred[0,:,0:int(TEST_SUPPORT_SET_NUM*0.5)] + eps
            patch_abnormal_score = pred[0,:,int(TEST_SUPPORT_SET_NUM*0.5):]

            patch_score =  patch_abnormal_score.mean(axis=1) / patch_normal_score.mean(axis=1)
            
   
            patch_score = patch_score.cpu().numpy()


            patch_score = patch_score.reshape((feat_H,feat_W))
            
            anomaly_map_resized = cv2.resize(patch_score, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE)) 
            
            anomaly_map_resized_blur = gaussian_filter(anomaly_map_resized, sigma=4)
            

            x = inv_normalize(x)
            input_x = cv2.cvtColor(x.permute(0,2,3,1).cpu().detach().numpy()[0]*255, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(img_save_base_path,
                                     '{}_a{}.png'.format(i,'img')), input_x)
            anomaly_map_norm = min_max_norm(anomaly_map_resized_blur)
            heatmap = cvt2heatmap(anomaly_map_norm*255)
            cv2.imwrite(os.path.join(img_save_base_path,
                                     '{}_b{}.png'.format(i,'map')), heatmap)
        
            if label_lst[i]:
                gt_map = cv2.imread(str(gt_path_lst[i]))
                gt_map = cv2.resize(gt_map, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE)) 
                gt_map = cv2.cvtColor(gt_map,cv2.COLOR_RGB2GRAY)
                
                cv2.imwrite(os.path.join(img_save_base_path,
                                         '{}_c{}.png'.format(i,'gt')), gt_map)
            else:
                gt_map = np.zeros([224, 224])

            label = label_lst[i]
            score = patch_score.max()
            
            
            
            if gt_map.max()!=0:
                aupro_list.append(compute_pro(np.expand_dims(gt_map, axis=0).astype(bool).astype(int),
                                              anomaly_map_resized_blur[np.newaxis,:,:]))
                
            
            gt_list_px_lvl.extend(gt_map.astype(bool).astype(int).ravel())
            pred_list_px_lvl.extend(anomaly_map_resized_blur.ravel())
            
            gt_list_img_lvl.append(label)
            pred_list_img_lvl.append(score)
            
            data_out_dic[i] = {
                'label':label,
                }
        
        print("\nTotal pixel-level auc-roc score :")
        pixel_auc = roc_auc_score(gt_list_px_lvl, pred_list_px_lvl)
        print(pixel_auc)
        
        print("Total image-level auc-roc score :")
        img_auc = roc_auc_score(gt_list_img_lvl, pred_list_img_lvl)
        print(img_auc)
        
        with open('./log/data_out_dic.json','w') as f:
            f.write(json.dumps(data_out_dic))
    
    return pixel_auc, img_auc, round(np.mean(aupro_list),3)






def compute_pro(masks: ndarray, amaps: ndarray, num_th: int = 200) -> None:

    """Compute the area under the curve of per-region overlaping (PRO) and 0 to 0.3 FPR
    Args:
        category (str): Category of product
        masks (ndarray): All binary masks in test. masks.shape -> (num_test_data, h, w)
        amaps (ndarray): All anomaly maps in test. amaps.shape -> (num_test_data, h, w)
        num_th (int, optional): Number of thresholds
    """
    assert isinstance(amaps, ndarray), "type(amaps) must be ndarray"
    assert isinstance(masks, ndarray), "type(masks) must be ndarray"
    assert amaps.ndim == 3, "amaps.ndim must be 3 (num_test_data, h, w)"
    assert masks.ndim == 3, "masks.ndim must be 3 (num_test_data, h, w)"
    assert amaps.shape == masks.shape, "amaps.shape and masks.shape must be same"
    
    assert set(masks.flatten()) == {0, 1}, "set(masks.flatten()) must be {0, 1}"
    assert isinstance(num_th, int), "type(num_th) must be int"

    df = pd.DataFrame([], columns=["pro", "fpr", "threshold"])
    binary_amaps = np.zeros_like(amaps, dtype=bool)

    min_th = amaps.min()
    max_th = amaps.max()
    delta = (max_th - min_th) / num_th

    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th] = 0
        binary_amaps[amaps > th] = 1

        pros = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                axes0_ids = region.coords[:, 0]
                axes1_ids = region.coords[:, 1]
                tp_pixels = binary_amap[axes0_ids, axes1_ids].sum()
                pros.append(tp_pixels / region.area)

        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()

        # df = df.append({"pro": mean(pros), "fpr": fpr, "threshold": th}, ignore_index=True)
        df = pd.concat([df, pd.DataFrame({"pro": [mean(pros)], "fpr": [fpr], "threshold": [th]})], ignore_index=True)

    # Normalize FPR from 0 ~ 1 to 0 ~ 0.3
    df = df[df["fpr"] < 0.3]
    df["fpr"] = df["fpr"] / df["fpr"].max()

    pro_auc = auc(df["fpr"], df["pro"])
    return pro_auc
