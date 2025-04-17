from loss_functions import psnr_score, temporal_contrastive_loss, optical_flow_consistency_loss, ssim_score, smooth_loss
from datasetloader import AVPerceptionDataset, FastDataLoader
from lnn_model import FlowEnhancedLNN
from collections import defaultdict
from tqdm import tqdm
import torch
import torch.nn.functional as F
import time
import os
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.transforms import ToPILImage, Resize, ToTensor
import random
import matplotlib.pyplot as plt


def visualize_prediction(pred, target, epoch, save_dir="visuals"):
    import os
    os.makedirs(save_dir, exist_ok=True)

    pred_np = pred[0].permute(1, 2, 0).cpu().detach().numpy()
    target_np = target[0].permute(1, 2, 0).cpu().detach().numpy()
    pred_np = np.clip(pred_np, 0, 1)
    target_np = np.clip(target_np, 0, 1)

    # Plot
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(target_np)
    plt.title("Ground Truth")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(pred_np)
    plt.title("Predicted Frame")
    plt.axis("off")

    # Save
    path = os.path.join(save_dir, f"epoch_{epoch:03d}.png")
    plt.savefig(path)
    plt.close()


def get_module(model):
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def train_epoch(model, dataloader, optimizer, device):
    model.train()
    metrics = defaultdict(float)
    count = 0

    for batch in tqdm(dataloader):
        if batch is None:
            continue
        frames, flow_gt = batch
        frames = frames.to(device)
        flow_gt = flow_gt.to(device)
        target = frames[:, -1]  # next frame
        target = target.clamp(0, 1)

        optimizer.zero_grad()
        pred_frame, pred_flow = model(frames)
        pred_frame = pred_frame.clamp(0, 1)
        pred_flow = pred_flow.clamp(0, 1)

        # losses
        loss_recon = F.mse_loss(pred_frame, target)
        psnr = psnr_score(pred_frame, target)
        ssim = torch.tensor(0.0)
        ssim = ssim_score(pred_frame, target)

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

        total_loss = loss_recon + contrastive + flow_loss + smooth
        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        metrics['total'] += total_loss.item()
        metrics['psnr'] += psnr.item()
        metrics['ssim'] += ssim.item()
        metrics['contrastive'] += contrastive.item()
        metrics['flow'] += flow_loss.item()
        metrics['smooth'] += smooth.item()
        metrics['recon_loss'] += loss_recon.item()
        count += 1

    for k in metrics:
        metrics[k] /= count
    return metrics


def main():

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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

    frame_root = "transformed_frames/"
    flow_root = "optical_flow/"
    split = "train/"

    transform = transforms.Compose([
        ToPILImage(),
        Resize(64),
        ToTensor()])

    dataset = AVPerceptionDataset(
        frame_root=frame_root,
        flow_root=flow_root,
        split=split,
        seq_length=5,
        transform=transform)
    dataloader = FastDataLoader(
        dataset, batch_size=160, shuffle=True, num_workers=6, pin_memory=True)

    model = FlowEnhancedLNN()
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    history = defaultdict(list)
    num_epochs = 30

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

        if (epoch + 1) % 3 == 0:
            torch.cuda.empty_cache()

        if (epoch + 1) % 2 == 0:
            model.eval()
            with torch.no_grad():
                for frames, _ in dataloader:
                    frames = frames.to(device)
                    target = frames[:, -1]
                    pred_frame, _ = model(frames)
                    pred_frame = pred_frame.clamp(0, 1)
                    target = target.clamp(0, 1)
                    visualize_prediction(pred_frame, target, epoch + 1)
                    break

    plt.figure(figsize=(12, 6))
    plt.plot(history['total'], label='Total Loss')
    plt.plot(history['psnr'], label='PSNR')
    plt.plot(history['ssim'], label='SSIM')
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Training Progress (LNN)")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"LNN_training_plot.png")
    print(f"📈 Plot saved as LNN_training_plot.png")
    plt.show()

    final_ckpt_path = "LNN_model.pth"
    torch.save(history, f"LNN_history.pth")
    torch.save(model, final_ckpt_path)
    print(f"Full model saved to {final_ckpt_path}")


if __name__ == "__main__":
    main()
