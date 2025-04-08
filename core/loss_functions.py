import torch
import torch.nn.functional as F
import pytorch_msssim

def psnr_loss(pred, target):
    mse = F.mse_loss(pred, target)
    psnr = -10 * torch.log10(mse + 1e-8)
    return -psnr

def temporal_contrastive_loss(features):
    """
    Contrastive loss for temporal stability using cosine similarity.
    Assumes `features` is a tensor of shape [B, T, D]
    """
    loss = 0.0
    B, T, D = features.shape
    for t in range(1, T):
        pos_sim = F.cosine_similarity(features[:, t], features[:, t-1], dim=1)
        loss += 1 - pos_sim.mean()  # minimize (1 - cosine similarity)
    return loss / (T - 1)

def optical_flow_consistency_loss(flow_pred, flow_gt):
    return F.l1_loss(flow_pred, flow_gt)

def ssim_loss(pred, target):
    return 1 - pytorch_msssim.ssim(pred, target, data_range=1.0, size_average=True)

def smooth_loss(flow):
    """
    Computes a smoothness loss for an optical flow tensor.
    
    Args:
        flow (Tensor): Optical flow tensor of shape [B, 2, H, W],
                       where 2 corresponds to the horizontal and vertical components.
                       
    Returns:
        Tensor: Smoothness loss (scalar).
    """
    # Compute the differences between adjacent pixels in the height dimension (vertical gradients)
    dy = torch.abs(flow[:, :, 1:, :] - flow[:, :, :-1, :])
    # Compute the differences between adjacent pixels in the width dimension (horizontal gradients)
    dx = torch.abs(flow[:, :, :, 1:] - flow[:, :, :, :-1])
    
    # Return the mean of these differences as the smoothness loss
    return torch.mean(dx) + torch.mean(dy)