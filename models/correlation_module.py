# -*- coding: utf-8 -*-
"""
https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/blob/master/pytorch_classification/vision_transformer/vit_model.py
"""
import torch
import torch.nn as nn



def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)




class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x



class Attention(nn.Module):
    def __init__(self,
                 dim,  
                 phase = None,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.phase = phase
        
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        
        self.q_proj = nn.Linear(dim, dim , bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim , bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim , bias=qkv_bias)
        
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x_q, x_s):
        B, N_q, C = x_q.shape
        _, N_s, _ = x_s.shape

        q = self.q_proj(x_q).reshape(B, N_q, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.q_proj(x_s).reshape(B, N_s, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.q_proj(x_s).reshape(B, N_s, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N_q, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    

class Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 phase = None,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                  act_layer=nn.GELU, 
                  # act_layer=nn.ReLU,
                 norm_layer=nn.LayerNorm
                 ):
        super(Block, self).__init__()
        
        self.norm1 = norm_layer(dim)

        self.attn = Attention(dim,
                              phase = phase,
                              num_heads=num_heads, 
                              qkv_bias=qkv_bias, 
                              qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, 
                              proj_drop_ratio=drop_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self, x_q, x_s):
        x_q_re = self.attn(x_q, x_s)
        x = x_q + x_q_re
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x + x_q
        return x



class CorreModule(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 
                 ):
        super(CorreModule, self).__init__()

        self.block11 = Block(dim=dim, num_heads = num_heads, mlp_ratio = mlp_ratio)
        self.block12 = Block(dim=dim, num_heads = num_heads, mlp_ratio = mlp_ratio)
        
        self.block21 = Block(dim=dim, num_heads = num_heads, mlp_ratio = mlp_ratio)
        self.block22 = Block(dim=dim, num_heads = num_heads,mlp_ratio = mlp_ratio)
        
        self.block31 = Block(dim=dim, num_heads = num_heads, mlp_ratio = mlp_ratio)
        self.block32 = Block(dim=dim, num_heads = num_heads, mlp_ratio = mlp_ratio)
        
        self.block41 = Block(dim=dim, num_heads = num_heads, mlp_ratio = mlp_ratio)
        self.block42 = Block(dim=dim, num_heads = num_heads, mlp_ratio = mlp_ratio)
        
        self.block51 = Block(dim=dim, num_heads = num_heads, mlp_ratio = mlp_ratio)
        self.block52 = Block(dim=dim, num_heads = num_heads, mlp_ratio = mlp_ratio)
        
        self.block61 = Block(dim=dim, num_heads = num_heads, mlp_ratio = mlp_ratio)
        self.block62 = Block(dim=dim, num_heads = num_heads, mlp_ratio = mlp_ratio)
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)
        self.mlp2 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
    def forward(self, query_feat, support_feat):
        

        query_feat_1 = self.block11(query_feat, support_feat)
        support_feat_1 = self.block12(support_feat, query_feat)
        scale = 1 / (query_feat_1.shape[-1] ** 0.5)*0.01
        pred1 = query_feat_1 @ support_feat_1.permute(0,2,1) * scale   

        query_feat_2 = self.block21(query_feat_1, support_feat_1)
        support_feat_2 = self.block22(support_feat_1, query_feat_1)
        pred2 = query_feat_2 @ support_feat_2.permute(0,2,1) * scale   

        query_feat_3 = self.block31(query_feat_2, support_feat_2)
        support_feat_3 = self.block32(support_feat_2, query_feat_2)
        pred3 = query_feat_3 @ support_feat_3.permute(0,2,1) * scale   

        query_feat_4 = self.block41(query_feat_3, support_feat_3)
        support_feat_4 = self.block42(support_feat_3, query_feat_3)
        pred4 = query_feat_4 @ support_feat_4.permute(0,2,1) * scale   

        query_feat_5 = self.block51(query_feat_4, support_feat_4)
        support_feat_5 = self.block52(support_feat_4, query_feat_4)
        pred5 = query_feat_5 @ support_feat_5.permute(0,2,1) * scale   
        
        return pred1, pred2, pred3, pred4, pred5



if __name__ == "__main__":
    BATCH_SIZE = 4
    N_QUERY = 196
    N_SUPPORT = 64
    INPUT_C = 256
    
    NUM_HEADS = 8
    MLP_RATIO = 4
    
    query_feat = torch.rand(BATCH_SIZE, N_QUERY, INPUT_C)
    support_feat = torch.rand(BATCH_SIZE, N_SUPPORT, INPUT_C)
    
    corre_module = CorreModule(dim=INPUT_C, num_heads=NUM_HEADS, mlp_ratio=MLP_RATIO)
    
    query_feat_out, support_feat_out = corre_module(query_feat,support_feat)
    
    print('query_feat_out:', query_feat_out.shape)
    print('support_feat_out:', support_feat_out.shape)
    
    
    

