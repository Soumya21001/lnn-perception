import torch
import torch.nn.functional as F
import pytorch_msssim
import torch
import torch.nn.functional as F


def psnr_score(pred, target, max_val=1.0):
    """
    Computes negative PSNR (to minimize).
    Assumes pred and target are in [0, 1].
    """
    pred = pred.clamp(0, 1)
    target = target.clamp(0, 1)
    mse = F.mse_loss(pred, target)
    psnr = 20 * torch.log10(max_val / torch.sqrt(mse + 1e-8))
    return psnr


def temporal_contrastive_loss(features):
    """
    Contrastive loss for temporal feature smoothness.
    Input: features of shape [B, T, D]
    """
    B, T, D = features.shape
    loss = 0.0
    for t in range(1, T):
        sim = F.cosine_similarity(
            features[:, t], features[:, t-1], dim=1)  # [B]
        # We want features to be temporally consistent
        loss += (1 - sim.mean())
    return loss / (T - 1)


def optical_flow_consistency_loss(flow_pred, flow_gt):
    """
    L1 loss between predicted and pseudo ground-truth optical flow.
    Assumes both are shaped [B, T-1, 2, H, W] or [B, 2, H, W].
    """
    # return torch.nn.MSELoss(flow_pred, flow_gt)
    epe = torch.norm(flow_pred - flow_gt, dim=2)
    return epe.mean()


def ssim_score(pred, target):
    pred = pred.clamp(0, 1)
    target = target.clamp(0, 1)
    return pytorch_msssim.ssim(pred, target, data_range=1.0, size_average=True)


def smooth_loss(flow):
    """
    Computes smoothness regularization on optical flow.
    Accepts [B, 2, H, W] or [B, T, 2, H, W]
    """
    if flow.dim() == 5:  # [B, T, 2, H, W]
        B, T, C, H, W = flow.shape
        flow = flow.view(B * T, C, H, W)  # Merge batch & time

    dx = torch.abs(flow[:, :, :, 1:] - flow[:, :, :, :-1])  # Horizontal
    dy = torch.abs(flow[:, :, 1:, :] - flow[:, :, :-1, :])  # Vertical

    return torch.mean(dx) + torch.mean(dy)
