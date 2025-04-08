import os
import torch
import torch.nn.functional as F
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
from lnn_model import FlowEnhancedLNN  
from loss_functions import psnr_loss, ssim_loss, temporal_contrastive_loss, optical_flow_consistency_loss, smooth_loss
from train import get_module     
from datasetloader import AVPerceptionDataset 
from datasetloader import FastDataLoader 
from torchvision import transforms
from torchvision.transforms import ToPILImage, Resize, ColorJitter, ToTensor    

transform = transforms.Compose([
        ToPILImage(),
        Resize((32, 32)),
        ColorJitter(0.2, 0.2),
        ToTensor()])

def evaluate(model, dataloader, device, save_vis=False, vis_dir='./eval_vis'):
    model.eval()
    metrics = defaultdict(float)
    count = 0
    vis_pred_frames = None
    vis_target_frames = None

    with torch.no_grad():
        for batch_idx, (frames, flow_gt) in enumerate(tqdm(dataloader, desc="Evaluating")):
            frames = frames.to(device)
            flow_gt = flow_gt.to(device)
            target = frames[:, -1]  # next frame is target

            pred_frame, pred_flow = model(frames)
            
            # Computing losses/metrics
            loss_recon = F.mse_loss(pred_frame, target)
            psnr = psnr_loss(pred_frame, target)
            ssim = ssim_loss(pred_frame, target)
            
            features = []
            core_model = get_module(model)
            for t in range(frames.size(1)):
                f = core_model.feature_extractor(frames[:, t])
                features.append(f)
            features = torch.stack(features, dim=1)  # [B, T, D]
            contrastive = temporal_contrastive_loss(features)
            
            flow_loss = optical_flow_consistency_loss(pred_flow, flow_gt)
            smooth = smooth_loss(pred_flow)
            
            total_loss = loss_recon + contrastive + flow_loss
            metrics['total'] += total_loss.item()
            metrics['psnr'] += psnr.item()
            metrics['ssim'] += ssim.item()
            metrics['contrastive'] += contrastive.item()
            metrics['flow'] += flow_loss.item()
            metrics['smooth'] += smooth.item()
            metrics['recon_loss'] += loss_recon.item()
            count += 1

            if save_vis and batch_idx == 0:
                vis_pred_frames = pred_frame[:2].cpu()
                vis_target_frames = target[:2].cpu()

    # Average metrics
    for k in metrics:
        metrics[k] /= count
    return metrics, vis_pred_frames, vis_target_frames

def visualize_results(pred_frames, target_frames, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    num_samples = pred_frames.size(0)
    for i in range(num_samples):
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(pred_frames[i].permute(1, 2, 0).clamp(0, 1).numpy())
        plt.title("Predicted Frame")
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.imshow(target_frames[i].permute(1, 2, 0).clamp(0, 1).numpy())
        plt.title("Target Frame")
        plt.axis("off")
        out_path = os.path.join(save_dir, f"sample_{i}.png")
        plt.savefig(out_path)
        plt.close()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    model = FlowEnhancedLNN()
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.to(device)

    checkpoint_path = "LNN_model_final.pth"
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print("Loaded checkpoint from", checkpoint_path)
    else:
        print("No checkpoint found. Evaluating untrained model.")

    frame_root = "BDD100K/transformedframes"
    flow_root = "BDD100K/opticalflow"
    split = "val"  # or "test" as needed

    transform = transform 
    dataset = AVPerceptionDataset(frame_root=frame_root, flow_root=flow_root, split=split, seq_length=5, transform=transform)
    dataloader = FastDataLoader(dataset, batch_size=512, shuffle=False, num_workers=16, pin_memory=True, persistent_workers=True)

   
    metrics, vis_pred_frames, vis_target_frames = evaluate(model, dataloader, device, save_vis=True, vis_dir="./eval_vis")
    print("Evaluation Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

  
    if vis_pred_frames is not None and vis_target_frames is not None:
        visualize_results(vis_pred_frames, vis_target_frames, save_dir="./eval_vis")
        print("Visualizations saved in ./eval_vis")

if __name__ == "__main__":
    main()