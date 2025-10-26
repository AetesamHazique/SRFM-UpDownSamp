import torch.utils.data as data
import torch
import numpy as np
import os
from train_utils import *
from os import listdir
from torchvision import transforms
from functools import reduce
from os.path import join
from torch.autograd import Variable
from PIL import Image, ImageOps
import random
from random import randrange

def rescale_img(img_in, scale):
    size_in = img_in.size
    new_size_in = tuple([int(x * scale) for x in size_in])
    img_in = img_in.resize(new_size_in, resample=Image.BICUBIC)
    return img_in

def get_patch(img_in, img_tar, img_bic, patch_size, scale, ix=-1, iy=-1):
    (ih, iw) = img_in.size
    (th, tw) = (scale * ih, scale * iw)

    patch_mult = scale #if len(scale) > 1 else 1
    tp = patch_mult * patch_size
    ip = tp // scale

    if ix == -1:
        ix = random.randrange(0, iw - ip + 1)
    if iy == -1:
        iy = random.randrange(0, ih - ip + 1)

    (tx, ty) = (scale * ix, scale * iy)

    img_in = img_in.crop((iy,ix,iy + ip, ix + ip))
    img_tar = img_tar.crop((ty,tx,ty + tp, tx + tp))
    img_bic = img_bic.crop((ty,tx,ty + tp, tx + tp))
                
    info_patch = {
        'ix': ix, 'iy': iy, 'ip': ip, 'tx': tx, 'ty': ty, 'tp': tp}

    return img_in, img_tar, img_bic, info_patch

def augment(img_in, img_tar, img_bic, flip_h=True, rot=True):
    info_aug = {'flip_h': False, 'flip_v': False, 'trans': False}
    
    if random.random() < 0.5 and flip_h:
        img_in = ImageOps.flip(img_in)
        img_tar = ImageOps.flip(img_tar)
        img_bic = ImageOps.flip(img_bic)
        info_aug['flip_h'] = True

    if rot:
        if random.random() < 0.5:
            img_in = ImageOps.mirror(img_in)
            img_tar = ImageOps.mirror(img_tar)
            img_bic = ImageOps.mirror(img_bic)
            info_aug['flip_v'] = True
        if random.random() < 0.5:
            img_in = img_in.rotate(180)
            img_tar = img_tar.rotate(180)
            img_bic = img_bic.rotate(180)
            info_aug['trans'] = True
            
    return img_in, img_tar, img_bic, info_aug

 
class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, patch_size, upscale_factor, data_augmentation, transform=None):
        super(DatasetFromFolder, self).__init__()
        self.image_filenames = [join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]
        self.patch_size = patch_size
        self.upscale_factor = upscale_factor
        self.transform = transform
        self.data_augmentation = data_augmentation

    def __getitem__(self, index):
        target = load_img(self.image_filenames[index])
        
        input = target.resize((int(target.size[0]/self.upscale_factor),int(target.size[1]/self.upscale_factor)), Image.BICUBIC)       
        bicubic = rescale_img(input, self.upscale_factor)
        
        input, target, bicubic, _ = get_patch(input,target,bicubic,self.patch_size, self.upscale_factor)
        
        if self.data_augmentation:
            input, target, bicubic, _ = augment(input, target, bicubic)
        
        if self.transform:
            input = self.transform(input)
            bicubic = self.transform(bicubic)
            target = self.transform(target)
                
        return input, target, bicubic

    def __len__(self):
        return len(self.image_filenames)
    
class DatasetFromFolder1(data.Dataset):
    def __init__(self, image_dir):
        super(DatasetFromFolder1, self).__init__()
        self.input_image_filenames = [join(image_dir[0], x) for x in listdir(image_dir[0]) if is_image_file(x)]
        self.target_image_filenames = [join(image_dir[1], x) for x in listdir(image_dir[1]) if is_image_file(x)]

    def __getitem__(self, index):
        input = load_img(self.input_image_filenames[index])
        target = load_img(self.target_image_filenames[index])
        
        # bicubic = rescale_img(input, self.upscale_factor)
        
        #input, target, bicubic, _ = get_patch(input,target,bicubic,self.patch_size, self.upscale_factor)
        img = Image.open('path/to/image.jpg')
        to_tensor = transforms.ToTensor()
        #tensor_img = to_tensor(img)
        return to_tensor(input), to_tensor(target)

    def __len__(self):
        return len(self.input_image_filenames)

class DatasetFromFolderEval(data.Dataset):
    def __init__(self, lr_dir, upscale_factor, transform=None):
        super(DatasetFromFolderEval, self).__init__()
        self.image_filenames = [join(lr_dir, x) for x in listdir(lr_dir) if is_image_file(x)]
        self.upscale_factor = upscale_factor
        self.transform = transform

    def __getitem__(self, index):
        input = load_img(self.image_filenames[index])
        _, file = os.path.split(self.image_filenames[index])

        bicubic = rescale_img(input, self.upscale_factor)
        
        if self.transform:
            input = self.transform(input)
            bicubic = self.transform(bicubic)
            
        return input, bicubic, file
      
    def __len__(self):
        return len(self.image_filenames)

class DatasetFromFolderEval(data.Dataset):
    def __init__(self, lr_dir, upscale_factor, transform=None):
        super(DatasetFromFolderEval, self).__init__()
        self.image_filenames = [join(lr_dir, x) for x in listdir(lr_dir) if is_image_file(x)]
        self.upscale_factor = upscale_factor
        self.transform = transform

    def __getitem__(self, index):
        input = load_img(self.image_filenames[index])
        _, file = os.path.split(self.image_filenames[index])

        bicubic = rescale_img(input, self.upscale_factor)
        
        if self.transform:
            input = self.transform(input)
            bicubic = self.transform(bicubic)
            
        return input, bicubic, file
    
class ImagePairTrainDataset(data.Dataset):
    def __init__(self, root_dir, norm_flag=1, transform=None):
        super(ImagePairTrainDataset, self).__init__()
        
        self.root_dir = root_dir
        self.image_list = os.listdir(os.path.join(root_dir, "training_wf"))
        self.norm_flag=norm_flag
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        input_path = os.path.join(self.root_dir, "training_wf", img_name)
        target_path = os.path.join(self.root_dir, "training_gt", img_name)
        input_image = Image.open(input_path)
        target_image = Image.open(target_path)

        if self.transform:
            input_image = self.transform(input_image)
            target_image = self.transform(target_image)

        if self.norm_flag:
            input_image = prctile_norm(input_image)
            target_image = prctile_norm(target_image)
        else:
            #input_image = input_image/65535
            #target_image = target_image/65535
            pass
            
        return input_image, target_image
    
class ImagePairValidationDataset(data.Dataset):
    def __init__(self, root_dir, norm_flag=1, transform=None):
        super(ImagePairValidationDataset, self).__init__()
        
        self.root_dir = root_dir
        self.image_list = os.listdir(os.path.join(root_dir, "validate_wf"))
        self.norm_flag=norm_flag
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        input_path = os.path.join(self.root_dir, "validate_wf", img_name)
        target_path = os.path.join(self.root_dir, "validate_gt", img_name)

        # Manipulation of imput image
        input_image = Image.open(input_path)
        
        # Manipulation of imput image
        target_image = Image.open(target_path) 

        if self.transform:
            input_image = self.transform(input_image)
            target_image = self.transform(target_image)

        if self.norm_flag:
            input_image = prctile_norm(input_image)
            target_image = prctile_norm(target_image)
        else:
            #input_image = input_image/65535
            #target_image = target_image/65535
            pass
            
        return input_image, target_image
    
class ImagePairTestDataset(data.Dataset):
    def __init__(self, lr_dir, upscale_factor, flevel=1,norm_flag=1,transform=None):
        super(ImagePairTestDataset, self).__init__()
        self.image_filenames = [join(lr_dir+"level_{:02d}/".format(flevel), x)\
                                 for x in listdir(lr_dir+"level_{:02d}/".format(flevel)) if is_image_file(x)]
        self.upscale_factor = upscale_factor
        self.norm_flag=norm_flag
        self.transform = transform

    def __getitem__(self, index):
        print(self.image_filenames[index])
        input = load_gimg(self.image_filenames[index])
        _, file = os.path.split(self.image_filenames[index])

        bicubic = rescale_img(input, self.upscale_factor)
        
        if self.transform:
            input = self.transform(input)
            bicubic = self.transform(bicubic)

        if self.norm_flag:
            #input = prctile_norm_inf(input)
            input = prctile_norm(input)
            #bicubic = prctile_norm_inf(bicubic)
            bicubic = prctile_norm(bicubic)
        else:
            input = input/65535
            bicubic = bicubic/65535
            #pass

        return input, bicubic, file
    
    def __len__(self):
        return len(self.image_filenames)