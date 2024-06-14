# -*- coding: utf-8 -*-
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt 
import math
import random
from glob import glob


from einops import rearrange
import imgaug.augmenters as iaa


CUTOUT_ANOMALY_PROB = 0

def generate_perlin_noise_mask(size=(224,224), threshold=0.7, min_perlin_scale=1, max_perlin_scale=5):
    # define perlin noise scale
    perlin_scalex = 2 ** random.randint(min_perlin_scale, max_perlin_scale)
    perlin_scaley = 2 ** random.randint(min_perlin_scale, max_perlin_scale)
    
    # generate perlin noise
    perlin_noise = rand_perlin_2d_np((size[0],size[1]), (perlin_scalex, perlin_scaley))
    
    # apply affine transform
    rot = iaa.Affine(rotate=(-45, 45))
    
    perlin_noise = min_max_norm(perlin_noise)
    if random.random() > 0.5:  
        perlin_noise = rot(image=perlin_noise)

    # make a mask by applying threshold
    mask_noise = np.where (
        perlin_noise > threshold, 
        np.ones_like(perlin_noise), 
        np.zeros_like(perlin_noise)
    )
    
    return mask_noise

def lerp_np(x,y,w):
    fin_out = (y-x)*w + x
    return fin_out

def rand_perlin_2d_np(shape, res, fade=lambda t: 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3):
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]].transpose(1, 2, 0) % 1

    angles = 2 * math.pi * np.random.rand(res[0] + 1, res[1] + 1)
    gradients = np.stack((np.cos(angles), np.sin(angles)), axis=-1)

    tile_grads = lambda slice1, slice2: np.repeat(np.repeat(gradients[slice1[0]:slice1[1], slice2[0]:slice2[1]],d[0],axis=0),d[1],axis=1)
    dot = lambda grad, shift: (
                np.stack((grid[:shape[0], :shape[1], 0] + shift[0], grid[:shape[0], :shape[1], 1] + shift[1]),
                            axis=-1) * grad[:shape[0], :shape[1]]).sum(axis=-1)

    n00 = dot(tile_grads([0, -1], [0, -1]), [0, 0])
    n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0])
    n01 = dot(tile_grads([0, -1], [1, None]), [0, -1])
    n11 = dot(tile_grads([1, None], [1, None]), [-1, -1])
    t = fade(grid[:shape[0], :shape[1]])
    return math.sqrt(2) * lerp_np(lerp_np(n00, n10, t[..., 0]), lerp_np(n01, n11, t[..., 0]), t[..., 1])


def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    return (image-a_min)/(a_max - a_min)    




class AnomalyGenerator():
    def __init__(self,
                 category,
                 img_size=(224,224),
                 transparency_range=[0.15,1.],
                 texture_source_dir=None,
                 structure_grid_size = 8,
                 use_foreground = True,
                 ):
        self.category = category
        self.img_size = img_size
        self.transparency_range = transparency_range
        
        self.structure_grid_size = structure_grid_size
        if texture_source_dir:
            self.texture_source_file_list = glob(os.path.join(texture_source_dir,'*/*'))
    
        self.no_foreground_category_lst = ['leather', 'tile', 'carpet', 'wood', 'grid', 'zipper','bottle']
        self.read_saved_foreground_category_lst = ['cable', 'pill', 'toothbrush', 'transistor', 'capsule']
        self.inverse_foreground_category_lst = ['hazelnut', 'metal_nut']
        
        if self.category in self.read_saved_foreground_category_lst:
            self.foreground_mask_init()
            
        self.use_foreground = use_foreground
        
    def foreground_mask_init(self,):
        target_foreground_mask = cv2.imread('./foreground_mask/{}.bmp'.format(self.category),
                                            cv2.IMREAD_GRAYSCALE)
        target_foreground_mask = cv2.resize(target_foreground_mask, self.img_size)
        self.rd_target_foreground_mask = target_foreground_mask.astype(bool).astype(int)
        
        
        
    def generate_anomaly(self, img):
        perlin_noise_mask = generate_perlin_noise_mask(self.img_size)

        if self.use_foreground:
            if self.category in self.no_foreground_category_lst:
                mask = perlin_noise_mask
            else:
                if self.category in self.read_saved_foreground_category_lst:
                    target_foreground_mask = self.rd_target_foreground_mask
                elif self.category in self.inverse_foreground_category_lst:
                    target_foreground_mask = self.generate_target_foreground_mask(img=img)
                    target_foreground_mask = -(target_foreground_mask-1)
                else:
                    target_foreground_mask = self.generate_target_foreground_mask(img=img)
                mask = perlin_noise_mask * target_foreground_mask
        else:
            mask = perlin_noise_mask

        mask_expanded = np.expand_dims(mask, axis=2)

        cutout_anomaly = (- mask_expanded + 1) * img

        if random.uniform(0, 1) < CUTOUT_ANOMALY_PROB:
            return (cutout_anomaly.astype(np.uint8), mask)
        else:

            anomaly_source_img = self.anomaly_source(img=img)
            factor = np.random.uniform(*self.transparency_range, size=1)[0]
            anomaly_source_img = factor * (mask_expanded * anomaly_source_img) + (1 - factor) * (mask_expanded * img)
            anomaly_source_img = cutout_anomaly + anomaly_source_img
            return (anomaly_source_img.astype(np.uint8), mask)
    
    def generate_target_foreground_mask(self, img):
        # convert RGB into GRAY scale
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # generate binary mask of gray scale image
        _, target_background_mask = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        target_background_mask = target_background_mask.astype(bool).astype(int)

        # invert mask for foreground mask
        target_foreground_mask = -(target_background_mask - 1)
        
        return target_foreground_mask
    
    def anomaly_source(self, img):
        
        p = np.random.uniform()
        if p < 0.5:
            anomaly_source_img = self._texture_source()
        else:
            anomaly_source_img = self._structure_source(img=img)
        
        return anomaly_source_img
        
    def _texture_source(self) -> np.ndarray:
        idx = np.random.choice(len(self.texture_source_file_list))
        texture_source_img = cv2.imread(self.texture_source_file_list[idx])
        texture_source_img = cv2.cvtColor(texture_source_img, cv2.COLOR_BGR2RGB)
        texture_source_img = cv2.resize(texture_source_img,
                                        dsize=(self.img_size[1], self.img_size[0])).astype(np.float32)
        
        return texture_source_img
        
    def _structure_source(self, img):
        structure_source_img = self.rand_augment()(image=img)
        
        assert self.img_size[0] % self.structure_grid_size == 0, 'structure should be devided by grid size accurately'
        grid_w = self.img_size[1] // self.structure_grid_size
        grid_h = self.img_size[0] // self.structure_grid_size
        
        structure_source_img = rearrange(
            tensor  = structure_source_img, 
            pattern = '(h gh) (w gw) c -> (h w) gw gh c',
            gw      = grid_w, 
            gh      = grid_h
        )
        disordered_idx = np.arange(structure_source_img.shape[0])
        np.random.shuffle(disordered_idx)

        structure_source_img = rearrange(
            tensor  = structure_source_img[disordered_idx], 
            pattern = '(h w) gw gh c -> (h gh) (w gw) c',
            h       = self.structure_grid_size,
            w       = self.structure_grid_size
        ).astype(np.float32)
        
        return structure_source_img
    
    def rand_augment(self):
        augmenters = [
            iaa.GammaContrast((0.5,2.0),per_channel=True),
            iaa.MultiplyAndAddToBrightness(mul=(0.8,1.2),add=(-30,30)),
            iaa.pillike.EnhanceSharpness(),
            iaa.AddToHueAndSaturation((-50,50),per_channel=True),
            iaa.Solarize(0.5, threshold=(32,128)),
            iaa.Posterize(),
            iaa.Invert(),
            iaa.pillike.Autocontrast(),
            iaa.pillike.Equalize(),
            iaa.Affine(rotate=(-45, 45))
        ]

        aug_idx = np.random.choice(np.arange(len(augmenters)), 3, replace=False)
        aug = iaa.Sequential([
            augmenters[aug_idx[0]],
            augmenters[aug_idx[1]],
            augmenters[aug_idx[2]]
        ])
        
        return aug
    
    
    
    
    
if __name__=='__main__':
    DATASET_PATH = ''
    AGen = AnomalyGenerator(category='leather', img_size=(224,224),
                            texture_source_dir = '')
    
    
    img_path = ''
    img = cv2.imread(os.path.join(DATASET_PATH,img_path))
    img = cv2.resize(img,(224,224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    img, mask = AGen.generate_anomaly(img=img)
    
    plt.figure()
    plt.imshow(img,plt.cm.gray)
    plt.axis('off')
    plt.show()

    plt.figure()
    plt.imshow(mask,plt.cm.gray)
    plt.axis('off')
    plt.show()
    

    