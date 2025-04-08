from loss_functions import psnr_loss, temporal_contrastive_loss, optical_flow_consistency_loss,ssim_loss, smooth_loss
from datasetloader import AVPerceptionDataset, FastDataLoader
from lnn_model import FlowEnhancedLNN, LTCCell, FeatureExtractor
from collections import defaultdict
from tqdm import tqdm
import torch
import torch.nn.functional as F
import time
import os
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.transforms import ToPILImage, Resize, ColorJitter, ToTensor


def get_module(model):
    return model.module if isinstance(model, torch.nn.DataParallel) else model

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    metrics = defaultdict(float)
    count = 0

    for frames, flow_gt in tqdm(dataloader):
        frames = frames.to(device)
        flow_gt = flow_gt.to(device)
        target = frames[:, -1] #next frame

        optimizer.zero_grad()
        pred_frame, pred_flow = model(frames)
        
        #losses
        loss_recon = F.mse_loss(pred_frame, target)
        psnr = psnr_loss(pred_frame, target)
        ssim = torch.tensor(0.0)
        ssim = ssim_loss(pred_frame, target)
        
        features = []
        core_model = get_module(model)
        with torch.no_grad():
            for t in range(frames.size(1)):
                f = core_model.feature_extractor(frames[:, t])
                features.append(f)
                
        features = torch.stack(features, dim=1)  # shape: [B, T, D]
        contrastive = temporal_contrastive_loss(features)
        
        flow_loss = optical_flow_consistency_loss(pred_flow, flow_gt)
        smooth = smooth_loss(pred_flow)

        total_loss = loss_recon + contrastive + flow_loss
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        metrics['total'] += total_loss.item()
        metrics['psnr'] += psnr.item()
        metrics['ssim'] += ssim.item()
        metrics['contrastive'] += contrastive.item()
        metrics['flow'] += flow_loss.item()
        metrics['smooth'] += smooth.item()
        metrics['recon_loss'] +=loss_recon.item()
        count += 1

    for k in metrics:
        metrics[k] /= count
    return metrics


transform = transforms.Compose([
        ToPILImage(),
        Resize((32, 32)),
        ColorJitter(0.2, 0.2),
        ToTensor()])

def main():
 
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        device = torch.device("cuda")
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        if num_gpus > 1:
            print(f"Multiple GPUs detected ({num_gpus})")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS backend.")
    else:
        device = torch.device("cpu")
        print("No GPU found. Using CPU.")

    frame_root = "../../BDD100K/transformedframes"
    flow_root = "../../BDD100K/opticalflow"
    split = "train"  # Can change to 'val' or 'test' if needed

    transform = transforms.Compose([
        ToPILImage(),
        Resize((32, 32)),
        ColorJitter(0.2, 0.2),
        ToTensor()])

    dataset = AVPerceptionDataset(
        frame_root=frame_root,
        flow_root=flow_root,
        split=split,
        seq_length=5,
        transform=transform)
    dataloader = FastDataLoader(dataset, batch_size=512, shuffle=True, num_workers=16, pin_memory=True, persistent_workers=True)
    
    
    model = FlowEnhancedLNN()
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)
    history = defaultdict(list)


    num_epochs = 10
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        start_time = time.time()

        metrics = train_epoch(model, dataloader, optimizer, device)
        duration = time.time() - start_time

        for k, v in metrics.items():
            history[k].append(v)

        print(f"Loss: {metrics['total']:.4f} | "
              f"Flow: {metrics['flow']:.4f} | "
              f"Smooth: {metrics['smooth']:.4f} | "
              f"PSNR: {metrics['psnr']:.4f} | "
              f"SSIM: {metrics['ssim']:.4f} | "
              f"Contrastive: {metrics['contrastive']:.4f} | "
              f"Recon: {metrics['recon_loss']:.4f}")
        print(f"Duration: {duration:.2f} sec")
        
    final_ckpt_path = "LNN_model.pth"
    torch.save(model, final_ckpt_path)
    print(f"Full model saved to {final_ckpt_path}")


    plt.figure(figsize=(10, 5))
    plt.plot(history['total'], label='Total Loss')
    plt.plot(history['psnr'], label='PSNR')
    plt.plot(history['contrastive'], label='Contrastive')
    plt.title("Training Metrics")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    


