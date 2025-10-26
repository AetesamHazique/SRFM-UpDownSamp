import torch
import torch.nn.functional as F

def calcPSNR(img1,img2,data_range=1.0):
    mse=F.mse_loss(img1,img2)
    psnr=20*torch.log10(data_range/torch.sqrt(mse))
    return psnr

import torch
import torch.nn as nn

class NormalizedRootMeanSquaredError(nn.Module):
    def __init__(self, normalization='mean'):
        super().__init__()
        self.normalization = normalization
        
    def forward(self, preds, target):
        # Flatten image dimensions (batch_size, C, H, W) -> (batch_size, C*H*W)
        preds_flat = preds.flatten(start_dim=1)
        target_flat = target.flatten(start_dim=1)
        
        # Calculate RMSE per sample
        squared_error = (preds_flat - target_flat) ** 2
        mse_per_sample = torch.mean(squared_error, dim=1)
        rmse_per_sample = torch.sqrt(mse_per_sample)
        
        # Calculate normalization factor
        if self.normalization == 'mean':
            denom = torch.mean(target_flat, dim=1)
        elif self.normalization == 'range':
            denom = torch.max(target_flat, dim=1).values - torch.min(target_flat, dim=1).values
        elif self.normalization == 'std':
            denom = torch.std(target_flat, dim=1)
        elif self.normalization == 'l2':
            denom = torch.norm(target_flat, p=2, dim=1)
        else:
            raise ValueError(f"Invalid normalization: {self.normalization}")
            
        # Avoid division by zero
        denom = torch.clamp(denom, min=1e-9)
        
        # Compute NRMSE and average across batch
        nrmse = rmse_per_sample / denom
        return torch.mean(nrmse)
