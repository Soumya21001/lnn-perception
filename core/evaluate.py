import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os
from torchvision import transforms
from torchvision.transforms import ToPILImage, Resize, ToTensor
from lnn_model import FlowEnhancedLNN
from datasetloader import AVPerceptionDataset, FastDataLoader
from loss_functions import psnr_score, ssim_score, optical_flow_consistency_loss, smooth_loss
from tqdm import tqdm


def visualize_prediction(pred, target, save_path):
    pred_np = pred[0].permute(1, 2, 0).cpu().detach().numpy()
    target_np = target[0].permute(1, 2, 0).cpu().detach().numpy()

    pred_np = np.clip(pred_np, 0, 1)
    target_np = np.clip(target_np, 0, 1)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(target_np)
    plt.title("Ground Truth")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(pred_np)
    plt.title("Predicted Frame")
    plt.axis("off")

    plt.savefig(save_path)
    plt.close()

def evaluate(model, dataloader, device, save_dir="lnn_eval_visuals", num_visuals=5):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    total_ssim, total_psnr, total_recon, total_flow, total_smooth = 0, 0, 0, 0, 0
    count = 0
    
    with torch.no_grad():
        for batch_idx, (frames, flow_gt) in enumerate(tqdm(dataloader)):
            frames = frames.to(device)
            flow_gt = flow_gt.to(device)
            target = frames[:, -1].clamp(0, 1)

            pred_frame, pred_flow = model(frames)
            pred_frame = pred_frame.clamp(0, 1)

            total_ssim += ssim_score(pred_frame, target).item()
            total_psnr += psnr_score(pred_frame, target).item()
            total_recon += F.mse_loss(pred_frame, target).item()
            total_flow += optical_flow_consistency_loss(pred_flow, flow_gt).item()
            total_smooth += smooth_loss(pred_flow).item()
            count += 1

            #Save N visuals
            if batch_idx < num_visuals :
                save_path = os.path.join(save_dir, f"lnn_eval_batch_{batch_idx+1}.png")
                visualize_prediction(pred_frame, target, save_path)

    print("\n Evaluation Metrics")
    print(f"SSIM Score:                {total_ssim / count:.4f}")
    print(f"PSNR Score:                {total_psnr / count:.4f}")
    print(f"Reconstruction Loss : {total_recon / count:.4f}")
    print(f"Flow Consistency Loss:     {total_flow / count:.4f}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Evaluating on:", device)

    transform = transforms.Compose([
        ToPILImage(),
        Resize(64),  
        ToTensor()])

    dataset = AVPerceptionDataset(
        frame_root="transformed_frames/",
        flow_root="optical_flow/",
        split="val",  # Change to "test" if needed
        seq_length=5,
        transform=transform
        )

    dataloader = FastDataLoader(dataset, batch_size=64, shuffle=False, num_workers=6, pin_memory=True)

    model = torch.load("LNN_model.pth", map_location=device, weights_only=False)
    
    model.to(device)

    evaluate(model, dataloader, device)

if __name__ == "__main__":
    main()