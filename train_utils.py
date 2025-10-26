import torch
import os
from os import listdir
from os.path import join
from PIL import Image, ImageOps
from torchvision import transforms
import numpy as np

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg",".tif"])

def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    #img = Image.open(filepath)
    #y, _, _ = img.split()
    return img

def load_gimg(filepath):
    img = Image.open(filepath)
    #img = Image.open(filepath)
    #y, _, _ = img.split()
    return img

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg",".tif"])

def prctile_norm(x, min_prc=0, max_prc=100):
    x=np.array(x) # retains original dtype and value range
    #x = torch.from_numpy(x)      # dtype matches NumPy array
    #x=x.numpy()
    y = (x-np.percentile(x, min_prc))/(np.percentile(x, max_prc)-np.percentile(x, min_prc)+1e-7)
    y[y > 1] = 1
    y[y < 0] = 0
    y=torch.from_numpy(y).to(torch.float32).unsqueeze(0)
    #y=torch.from_numpy(y).to(torch.float32).squeeze(0)
    #y=torch.from_numpy(y).to(torch.float32)
    return y

def prctile_norm_inf(x, min_prc=0, max_prc=100): # this def for inference
    x=np.array(x) # retains original dtype and value range
    x = torch.from_numpy(x)      # dtype matches NumPy array
    x=x.numpy()
    y = (x-np.percentile(x, min_prc))/(np.percentile(x, max_prc)-np.percentile(x, min_prc)+1e-7)
    y[y > 1] = 1
    y[y < 0] = 0
    y=torch.from_numpy(y).to(torch.float32).unsqueeze(0)
    #y=torch.from_numpy(y).to(torch.float32)
    return y

def mat2gray(A, amin=None, amax=None):
    """
    Scales a PyTorch tensor A to the range [0, 1], mimicking MATLAB's mat2gray.
    Optionally, specify amin and amax to set custom input bounds.
    """
    if amin is None:
        amin = torch.min(A)
    if amax is None:
        amax = torch.max(A)
    # Prevent division by zero
    if amax == amin:
        return torch.zeros_like(A, dtype=torch.float32)
    return ((A - amin) / (amax - amin)).to(torch.float32)

def prctile_norm_torch(x, min_prc=0, max_prc=100):
    
    min_val = torch.quantile(torch.tensor(x,dtype=torch.float32), min_prc / 100.0)
    max_val = torch.quantile(torch.tensor(x,dtype=torch.float32), min_prc/100.0)
    y = (x - min_val) / (max_val - min_val + 1e-7)
    y = torch.clamp(y, 0, 1)
    return y

def normalize_image(img, a=0.0, b=1.0):
    # Convert PIL image to tensor (values in [0,1])
    tensor1 = transforms.ToTensor()(img)
    tensor=torch.tensor(tensor1,dtype=torch.float16)
    min_val = tensor.amin()
    max_val = tensor.amax()
    # Avoid division by zero if image is flat
    if max_val == min_val:
        return torch.full_like(tensor, a)
    normalized = a + (tensor - min_val) * ((b - a) / (max_val - min_val))
    return normalized

def percentile_normalize(tensor, lower_percentile=1, upper_percentile=99):
    """
    Normalize a tensor based on lower and upper percentiles.
    Values below the lower percentile are set to 0, values above the upper percentile are set to 1.
    The rest are scaled linearly between 0 and 1.
    
    Args:
        tensor (torch.Tensor): Input tensor.
        lower_percentile (float): Lower percentile (0-100).
        upper_percentile (float): Upper percentile (0-100).
        
    Returns:
        torch.Tensor: Normalized tensor.
    """
    # Compute percentiles
    tensor_norm=transforms.functional.convert_image_dtype(tensor,torch.float32)
    #transforms.functional.convert_image_dtype(transforms.ToTensor(),torch.float32)
    lower = torch.quantile(tensor_norm, lower_percentile / 100.0)
    upper = torch.quantile(tensor_norm, upper_percentile / 100.0)
    
    # Clip and normalize
    tensor_clipped = torch.clamp(tensor_norm, min=lower, max=upper)
    normalized = (tensor_clipped - lower) / (upper - lower + 1e-8)  # add epsilon to avoid division by zero
    
    return normalized