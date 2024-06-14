# -*- coding: utf-8 -*-
import torch
from torch.nn import functional as F 
import numpy as np
import cv2
import random
import time

from utils import get_feat_set
from evaluate import evaluate


def training(
            TRAINING_STEPS,
            feature_extractor,
            corre_module,
            C ,H, W,
            normal_feat_bank,
            abnormal_feat_bank,
            QUERY_SET_NUM,
            SUPPORT_SET_NUM,
            LEARNING_RATE,
            test_data_loader,
            INPUT_IMAGE_SIZE,
            CATEGORY,
            TEST_SUPPORT_SET_NUM,
            test_support_feat,
            device,
            patch_avg_proto
             ):
    print('\ntraining...')
    corre_module.train()
    
    dic = {}
    dic['evaluate_step'] = []
    dic['pixel_auc'] = []
    dic['img_auc'] = []
    dic['aupro'] = []
    max_pixel_auc = 0
    max_image_auc = 0
    max_aupro = 0
    
    
    optimizer = torch.optim.SGD(params=corre_module.parameters(),
                                lr=LEARNING_RATE, 
                                momentum=0.95, 
                                nesterov=False,
                                weight_decay=1e-5
                                )
    
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1200,2400], gamma=0.5)
    
    total_normal_feat_num = normal_feat_bank.shape[0]
    spt_qry_normal_feat_bank_num = int(total_normal_feat_num*0.5)
    normal_sel_idx_lst = [i for i in range(total_normal_feat_num)]
    
    start_time = time.time()
    
    
    for step in range(TRAINING_STEPS):
        random.shuffle(normal_sel_idx_lst) 
        spt_normal_feat_sel_idx = normal_sel_idx_lst[0:spt_qry_normal_feat_bank_num]
        qry_normal_feat_sel_idx = normal_sel_idx_lst[spt_qry_normal_feat_bank_num:]
        
        spt_normal_feature_bank = normal_feat_bank[spt_normal_feat_sel_idx]
        qry_normal_feature_bank = normal_feat_bank[qry_normal_feat_sel_idx]
        
        support_feat, support_label, spt_set_normal_sel_index = get_feat_set(spt_normal_feature_bank, 
                                                        abnormal_feat_bank, 
                                                        SUPPORT_SET_NUM,
                                                        random_out = True,
                                                        )
        
        query_feat, query_label, qry_set_normal_sel_index = get_feat_set(qry_normal_feature_bank, 
                                                        abnormal_feat_bank, 
                                                        QUERY_SET_NUM,
                                                        random_out = True,
                                                        )
        

        support_feat = support_feat.unsqueeze(dim=0).to(device)
        query_feat = query_feat.unsqueeze(dim=0).to(device)
        
        
        ### cal label
        inv_query_label = 1 - query_label
        inv_support_label = 1 - support_label
        target_label_1 = query_label.unsqueeze(1) @ support_label.unsqueeze(0) 
        target_label_0 = inv_query_label.unsqueeze(1) @ inv_support_label.unsqueeze(0)
        target_label = target_label_0 + target_label_1
        target_label = target_label.unsqueeze(dim=0).to(device)
        
        
        pred_out_lst = corre_module(query_feat,support_feat)
        
        pred = torch.cat([pred.unsqueeze(dim=-1) for pred in pred_out_lst], dim=-1)  ## torch.Size([1, 196, 256, 5])
        pred = torch.sigmoid(torch.mean(pred,dim=-1))
        
        loss = F.binary_cross_entropy(pred, target_label) 
        

        if step % 20 == 0:
            print('step:', step, 
                  ' loss:', loss.item(), 
                  ' speed(second/step):', round((time.time() - start_time) / (step + 1),2)
                  )
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(corre_module.parameters(), 1.0)
        optimizer.step()
    
        if scheduler:
            scheduler.step()
        
        if (step+1) % 60 == 0:
            pixel_auc, img_auc, aupro =  evaluate(
                                        feature_extractor,
                                        corre_module,
                                        C,
                                        H,
                                        W,
                                        test_data_loader,
                                        INPUT_IMAGE_SIZE,
                                        CATEGORY,
                                        TEST_SUPPORT_SET_NUM,
                                        test_support_feat,
                                        device,
                                        patch_avg_proto,
                                        )
            dic['evaluate_step'].append(step+1)
            dic['pixel_auc'].append(pixel_auc)
            dic['img_auc'].append(img_auc)
            dic['aupro'].append(aupro)
            
            if (pixel_auc + img_auc) > (max_pixel_auc + max_image_auc):
                max_pixel_auc = pixel_auc
                max_image_auc = img_auc
                print('max_pixel_auc:',max_pixel_auc,'max_image_auc',max_image_auc)
                
            corre_module.train()
            
    return dic
    
    
    
    
    
    
