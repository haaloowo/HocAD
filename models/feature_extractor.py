# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.nn import functional as F 
from .resnet import resnet18


class FeatureFusionBlk(nn.Module):
    def __init__(self, feature_resolution=14):
        super(FeatureFusionBlk, self).__init__()
        self.blk = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            
        )
        self.feature_resolution = feature_resolution

    def forward(self, x1, x2, x3):
        ### layer 2 3
        x2_resize = F.interpolate(x2, size=[self.feature_resolution, self.feature_resolution], mode='bilinear')
        y = torch.cat([x2_resize, x3], dim=1)
        
        # ### layer 1 2
        # x1_resize = F.interpolate(x1, size=[self.feature_resolution, self.feature_resolution], mode='bilinear')
        # x2_resize = F.interpolate(x2, size=[self.feature_resolution, self.feature_resolution], mode='bilinear')
        # y = torch.cat([x1_resize, x2_resize], dim=1)
        
        
        ### layer 1 2 3
        # x1_resize = F.interpolate(x1, size=[self.feature_resolution, self.feature_resolution], mode='bilinear')
        # x2_resize = F.interpolate(x2, size=[self.feature_resolution, self.feature_resolution], mode='bilinear')
        # y = torch.cat([x1_resize, x2_resize, x3], dim=1)
        return y





class FeatureExtractor():
    
    def __init__(self, device, model_name = 'resnet18', multilayer = False):
        
        self.feature_extractor, _ = resnet18(pretrained=True)
        
        self.feature_extractor = self.feature_extractor.to(device)
        self.feature_extractor.eval()
        
        self.feature_fusion = FeatureFusionBlk()
        self.multilayer = multilayer
    
    def get_feature_extractor(self):
        return self.feature_extractor
    
    def get_feature(self,x):
        features = self.feature_extractor(x) 
        if self.multilayer:
            feature = self.feature_fusion(features[0], features[1], features[2])
        else:
            ### layer 1
            feature = F.interpolate(features[0], size=[14, 14], mode='bilinear')
            ## layer 2
            # feature = F.interpolate(features[1], size=[14, 14], mode='bilinear')
            ### layer 3
            # feature = features[2]
        return feature
    
    
    

