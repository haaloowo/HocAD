# -*- coding: utf-8 -*-

import torch
import random
import os
import argparse

from models import FeatureExtractor, CorreModule
from training import training
from utils import MVTecDataset, gen_abnormal_feat_bank
from utils import get_test_feat_set
from utils import AnomalyGenerator, random_sel_img_and_read, get_feature_extractor_out_size
from utils import gen_normal_feat_bank, save_score_data

SEED = 6
random.seed(SEED)


rotate_category_lst = ['bottle', 'carpet', 'grid', 'hazelnut', 
                       'leather', 'metal_nut', 'screw', 'tile', 'wood']
train_steps_dic = {
    
    ### MVTec
    'bottle':800,    'hazelnut':800,    'tile':600,
    'cable':1200,    'leather':400,     'toothbrush':800,
    'capsule':1500,  'metal_nut':1200,  'transistor':1200,
    'carpet':1200,   'pill':1500,       'wood':800,
    'grid':900,      'screw':900,       'zipper':600,
    
    ### MPDD
    'bracket_black':1200, 
    'bracket_brown':1200, 
    'bracket_white':1200, 
    'connector':1200, 
    'metal_plate':1200, 
    'tubes':1200, 
    
    }



if __name__=='__main__':
    
    DATASET_PATH = '/media/yf/CODE/Dataset/MVTec'
    TEXTURE_SOURCE_DIR = '/media/yf/CODE/Dataset/dtd/images'
    
    parser = argparse.ArgumentParser(description='ANOMALDETECTION')
    parser.add_argument('--CATEGORY', default='screw')
    args = parser.parse_args()
    
    CATEGORY = args.CATEGORY

    SHOT_NUM = 2
    
    USE_MULTI_LAYER_FEATURE = True
    
    PATCH_ABNORMAL_THRESHOLD_MIN = 50
    PATCH_ABNORMAL_THRESHOLD_MAX = 200
    
    
    LEARNING_RATE = 5e-3
    
    TEST_SUPPORT_SET_NUM = 392 * SHOT_NUM   

    SUPPORT_SET_NUM = 256  
    QUERY_SET_NUM = 196   
    
    INPUT_IMAGE_SIZE = 224
    PATCH_SIZE = 16
    PATCH_NUM_AXIS = 14
    PATCH_NUM = PATCH_NUM_AXIS * PATCH_NUM_AXIS
    
    TRAINING_STEPS = train_steps_dic[CATEGORY] 
    
    if SHOT_NUM == 2 or SHOT_NUM == 4:
        TOTAL_ABNORMAL_FEAT_NUM = 100000   
    else:
        TOTAL_ABNORMAL_FEAT_NUM = 150000
    
    if CATEGORY in rotate_category_lst:   
        INPUT_EXPAND_ROTATE = True
    else:
        INPUT_EXPAND_ROTATE = False

        
        
    if not os.path.exists('./log'):
        os.mkdir('./log')
    if not os.path.exists('./log/score'):
        os.mkdir('./log/score')
    if not os.path.exists('./log/output'):
        os.mkdir('./log/output')
    if not os.path.exists('./log/sim_abnormal_feat'):
        os.mkdir('./log/sim_abnormal_feat')
    if not os.path.exists('./log/sim_abnormal_feat/seed{}'.format(SEED)):
        os.mkdir('./log/sim_abnormal_feat/seed{}'.format(SEED))
        

    sim_abnormal_file = '{}-shot-thres{}'.format(SHOT_NUM, PATCH_ABNORMAL_THRESHOLD_MIN)
    
  
    sim_abnormal_file = sim_abnormal_file + '-diff'

    if not os.path.exists('./log/sim_abnormal_feat/seed{}/{}'.format(SEED, sim_abnormal_file)):
        os.mkdir('./log/sim_abnormal_feat/seed{}/{}'.format(SEED, sim_abnormal_file))
        
    
    score_save_path = os.path.join('./log/score/{}.json'.format(CATEGORY))
    abnormal_feat_save_path = './log/sim_abnormal_feat/seed{}/{}/{}.npy'.format(SEED, sim_abnormal_file, CATEGORY)
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    


    print('-----------------------------------------------')
    print('category:',CATEGORY)
    print('device: ',device)
    
    
    ### ---------------------------initialize-------------------------------------
    feature_extractor = FeatureExtractor(device, model_name = 'resnet18', multilayer = USE_MULTI_LAYER_FEATURE)
    C,H,W = get_feature_extractor_out_size(feature_extractor, INPUT_IMAGE_SIZE, device)

    module_dim = C * 2 

    corre_module = CorreModule(dim=module_dim, 
                               num_heads=8, 
                               mlp_ratio=1,
                               ).to(device)

    
    AGen = AnomalyGenerator(category=CATEGORY, 
                            img_size=(INPUT_IMAGE_SIZE,INPUT_IMAGE_SIZE),
                            texture_source_dir=TEXTURE_SOURCE_DIR,
                            use_foreground = True,
                            )
    anormaly_generator = AGen.generate_anomaly
    
    
    input_img_arr = random_sel_img_and_read(DATASET_PATH, CATEGORY, SHOT_NUM, INPUT_IMAGE_SIZE, 
                                            INPUT_EXPAND_ROTATE)
    TOTAL_IMG_NUM = input_img_arr.shape[0]  
    TOTAL_PATCH_NUM = TOTAL_IMG_NUM * PATCH_NUM


    test_data_loader = MVTecDataset(root=os.path.join(DATASET_PATH, CATEGORY), phase='test')
    
    
    ###-----------------------------data prepare---------------------------------
    normal_feat_bank, foreground_normal_feat_bank, foreground_patch_idx = gen_normal_feat_bank(
                                                                                        CATEGORY,
                                                                                        input_img_arr, 
                                                                                        feature_extractor,
                                                                                        H, W, C, 
                                                                                        device
                                                                                        )
    
    
    patch_avg_proto = normal_feat_bank.reshape(TOTAL_IMG_NUM, -1, C).mean(dim=0)  ## torch.Size([196, 384])
    patch_avg_proto_repeat = torch.cat([patch_avg_proto for i in range(TOTAL_IMG_NUM)], dim=0)

    diff_feat = torch.abs(normal_feat_bank - patch_avg_proto_repeat)
    normal_feat_bank = torch.cat([normal_feat_bank, diff_feat], dim=-1)
    print('normal_feature_bank:',normal_feat_bank.shape)

    
    abnormal_feat_bank = gen_abnormal_feat_bank(TOTAL_ABNORMAL_FEAT_NUM,
                                                C,
                                                input_img_arr,
                                                PATCH_ABNORMAL_THRESHOLD_MIN,
                                                PATCH_ABNORMAL_THRESHOLD_MAX,
                                                feature_extractor,
                                                anormaly_generator,
                                                abnormal_feat_save_path,
                                                device,
                                                SAVE_ABNORMAL_FEAT = True,
                                                USE_DIFF_FEATURE = True,
                                                patch_avg_proto_feat = patch_avg_proto
                                                )
    print('abnormal_feat_bank:', abnormal_feat_bank.shape)
    
    test_support_feat, _ = get_test_feat_set(
                                            normal_feat_bank, 
                                            abnormal_feat_bank, 
                                            TEST_SUPPORT_SET_NUM
                                            )
    print('test_support_feat:', test_support_feat.shape)
    


    dic = training(
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
                    )

    save_score_data(score_save_path,dic)
        





