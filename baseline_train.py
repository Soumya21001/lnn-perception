import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from tqdm import tqdm
from torchvision import transforms
from torchvision.transforms import ToPILImage, Resize, ToTensor
import matplotlib.pyplot as plt
from core.datasetloader import AVPerceptionDataset, FastDataLoader
from core.loss_functions import psnr_score, ssim_score, temporal_contrastive_loss, optical_flow_consistency_loss, smooth_loss
from baselines import CNNBaseline, CNNLSTMBaseline  # You should define these
from core.train import visualize_prediction



# ----- Select model type -----
# ----- Ask user for model type -----
print("Choose baseline model to train:")
print("1 - CNNBaseline")
print("2 - CNNLSTMBaseline")

choice = input("Enter 1 or 2: ").strip()

if choice == "1":
    BASELINE_TYPE = "cnn"
elif choice == "2":
    BASELINE_TYPE = "cnn_lstm"
else:
    raise ValueError("Invalid input! Please enter 1 or 2.")

# ----- Training loop -----
def train_epoch(model, dataloader, optimizer, device):
    model.train()
    metrics = defaultdict(float)
    count = 0

    for frames, flow_gt in tqdm(dataloader):
        if frames is None or flow_gt is None:
            continue

        frames = frames.to(device)
        flow_gt = flow_gt.to(device)
        target = frames[:, -1].clamp(0, 1)

        optimizer.zero_grad()
        pred_frame, pred_flow = model(frames)
        pred_frame = pred_frame.clamp(0, 1)
        pred_flow = pred_flow.clamp(0,1)

        loss_recon = F.mse_loss(pred_frame, target)
        psnr = psnr_score(pred_frame, target)
        ssim = ssim_score(pred_frame, target)

        with torch.no_grad():
            features = [model.feature_extractor(frames[:, t]) for t in range(frames.size(1))]
        features = torch.stack(features, dim=1)

        contrastive = temporal_contrastive_loss(features)
        flow_loss = optical_flow_consistency_loss(pred_flow, flow_gt)
        smooth = smooth_loss(pred_flow)

        total_loss = loss_recon + contrastive + flow_loss + smooth
        total_loss.backward()
        optimizer.step()

        # Track metrics
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

# ----- Main -----
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using", device)

    transform = transforms.Compose([
        ToPILImage(),
        Resize(64),
        ToTensor()
    ])

    dataset = AVPerceptionDataset("transformed_frames/", "optical_flow/", split='train', seq_length=5, transform=transform)
    dataloader = FastDataLoader(dataset, batch_size=128, shuffle=True, num_workers=8, pin_memory=True)

    if BASELINE_TYPE == "cnn":
        model = CNNBaseline()
    elif BASELINE_TYPE == "cnn_lstm":
        model = CNNLSTMBaseline()
    else:
        raise ValueError("Invalid baseline type")

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    history = defaultdict(list)

    for epoch in range(30):
        print(f"\nEpoch {epoch + 1}/30")
        metrics = train_epoch(model, dataloader, optimizer, device)

        for k, v in metrics.items():
            history[k].append(v)
            
        print(f"Loss: {metrics['total']:.4f} | "
              f"Flow: {metrics['flow']:.4f} | "
              f"Smooth: {metrics['smooth']:.4f} | "
              f"PSNR: {metrics['psnr']:.4f} | "
              f"SSIM: {metrics['ssim']:.4f} | "
              f"Contrastive: {metrics['contrastive']:.4f} | "
              f"Recon: {metrics['recon_loss']:.4f}")

        # print(f"Loss: {metrics['total']:.4f} | PSNR: {metrics['psnr']:.4f} | SSIM: {metrics['ssim']:.4f}")

        # Visualization every 2 epochs
        if (epoch + 1) % 2 == 0:
            model.eval()
            with torch.no_grad():
                for frames, _ in dataloader:
                    frames = frames.to(device)
                    target = frames[:, -1]
                    pred_frame, _ = model(frames)
                    visualize_prediction(pred_frame.clamp(0,1), target.clamp(0,1), epoch + 1, save_dir="baseline_{BASELINE_TYPE}/")
                    break

    # Save model
    torch.save(history, f"{BASELINE_TYPE}_history.pth")
    torch.save(model.state_dict(), f"{BASELINE_TYPE}_baseline.pth")
    print(f"Model saved to {BASELINE_TYPE}_baseline.pth")

    # Plot metrics
    plt.plot(history['total'], label="Total Loss")
    plt.plot(history['psnr'], label="PSNR")
    plt.plot(history['ssim'], label = 'SSIM')
    plt.legend()
    plt.grid(True)
    plt.title("Training Metrics")
    plt.show()
    plt.savefig(f"{BASELINE_TYPE}_training_plot.png")
    print(f"Plot saved as {BASELINE_TYPE}_training_plot.png")

if __name__ == "__main__":
    main()