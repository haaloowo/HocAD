# -*- coding: utf-8 -*-

import json
import numpy as np
import os 
import pandas as pd

category_lst = ['bottle', 'cable', 'capsule', 'carpet', 
                'grid', 'hazelnut', 'leather', 'metal_nut', 
                'pill', 'screw', 'tile', 'toothbrush', 
                'transistor', 'wood', 'zipper']
    
# category_lst = ['bracket_black', 'bracket_brown', 
#                 'bracket_white', 'connector', 
#                 'metal_plate', 'tubes']

texture_category_lst = ['leather', 'wood', 'carpet', 'tile', 'grid']

category_num = len(category_lst)



total_pixel_auc_score = 0
total_image_auc_score = 0
total_aupro = 0

texture_total_pixel_auc_score = 0
texture_total_image_auc_score = 0



object_total_pixel_auc_score = 0
object_total_image_auc_score = 0

data = [
    ['category', 'img auc', 'pxl auc', 'pro']
]

for category_name in category_lst:

    with open('./log/score/{}.json'.format(category_name),'r') as f:
        data_dic = json.loads(f.read())
        pixel_auc_lst = data_dic['pixel_auc']
        img_auc_lst = data_dic['img_auc']
        aupro_lst = data_dic['aupro']
    
    score_lst = []
    
    for i,pixel_auc in enumerate(pixel_auc_lst):
        img_auc = img_auc_lst[i]
        score_lst.append(pixel_auc + img_auc)
        
    score_arr = np.array(score_lst)
    max_idx = np.argmax(score_arr)
        
    pixel_auc_score = pixel_auc_lst[max_idx]
    image_auc_score = img_auc_lst[max_idx]
    aupro = aupro_lst[max_idx]
    
    print(round(image_auc_score,3), '  ',round(pixel_auc_score,3), aupro, '     {}  image-pixel  '.format(category_name))
    
    data.append([category_name, round(image_auc_score,3), round(pixel_auc_score,3), aupro])
    
    
    total_aupro += aupro
    total_pixel_auc_score += pixel_auc_score
    total_image_auc_score += image_auc_score
    
    if category_name in texture_category_lst:
        texture_total_pixel_auc_score += pixel_auc_score
        texture_total_image_auc_score += image_auc_score
    else:
        object_total_pixel_auc_score += pixel_auc_score
        object_total_image_auc_score += image_auc_score
    
print('\nall')
print('image level avg auc:', round(total_image_auc_score / category_num, 3))
print('pixel level avg auc:', round(total_pixel_auc_score / category_num, 3))
print('pro:', round(total_aupro / category_num,3))
print('\ntexture')
print('image level avg auc:', round(texture_total_image_auc_score / 5, 3))
print('pixel level avg auc:', round(texture_total_pixel_auc_score / 5, 3))
            
print('\nobject')
print('image level avg auc:', round(object_total_image_auc_score / 10, 3))
print('pixel level avg auc:', round(object_total_pixel_auc_score / 10, 3))
                
            





data.append(['avg', round(total_image_auc_score / 15, 4), round(total_pixel_auc_score / 15, 4), round(total_aupro / 15, 4)])


df = pd.DataFrame(data)

df.to_excel('./log/score.xlsx', index=False)
