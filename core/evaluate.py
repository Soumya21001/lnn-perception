import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import ToPILImage, Resize, ToTensor
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
import os

from datasetloader import AVPerceptionDataset, FastDataLoader
from loss_functions import (
    psnr_score,
    ssim_score,
    optical_flow_consistency_loss,
)
from baselines.baselines import CNNBaseline, CNNLSTMBaseline
from lnn_model import FlowEnhancedLNN


def visualize_prediction(pred, target, save_path):
    pred = pred[0].permute(1, 2, 0).cpu().numpy().clip(0, 1)
    target = target[0].permute(1, 2, 0).cpu().numpy().clip(0, 1)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(target)
    plt.title("Ground Truth")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(pred)
    plt.title("Predicted")
    plt.axis("off")

    plt.savefig(save_path)
    plt.close()


def evaluate(model, dataloader, device, save_dir):
    model.eval()
    metrics = defaultdict(float)
    count = 0

    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for batch_idx, (frames, flow_gt) in enumerate(tqdm(dataloader)):
            if frames is None or flow_gt is None:
                continue

            # Move data to device
            frames, flow_gt = frames.to(device), flow_gt.to(device)
            target = frames[:, -1].clamp(0, 1)

            # Forward pass
            pred_frame, pred_flow = model(frames)
            pred_frame = pred_frame.clamp(0, 1)
            pred_flow = pred_flow.clamp(0, 1)

            # Compute losses and metrics
            loss_recon = F.mse_loss(pred_frame, target)
            psnr = psnr_score(pred_frame, target)
            ssim = ssim_score(pred_frame, target)
            flow_loss = optical_flow_consistency_loss(pred_flow, flow_gt)

            # features = [
            #     model.feature_extractor(frames[:, t]) for t in range(frames.size(1))
            # ]
            # features = torch.stack(features, dim=1)

            # Model-specific metrics
            metrics["psnr"] += psnr.item()
            metrics["ssim"] += ssim.item()
            metrics["recon"] += loss_recon.item()
            metrics["flow"] += flow_loss.item()
            count += 1

            # Save visualizations for first N batches
            N = 5  # Number of visuals to save
            if batch_idx < N:
                save_path = os.path.join(
                    save_dir, f"eval_batch_{batch_idx+1}.png")
                visualize_prediction(pred_frame, target, save_path)

    for k in metrics:
        metrics[k] /= count

    print("\nEvaluation Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    transform = transforms.Compose([ToPILImage(), Resize(64), ToTensor()])

    dataset = AVPerceptionDataset(
        "transformed_frames/",
        "optical_flow/",
        split="val",
        seq_length=5,
        transform=transform,
    )
    dataloader = FastDataLoader(
        dataset, batch_size=64, shuffle=False, num_workers=6)

    model_dict = {
        CNNBaseline(): "cnn_baseline",
        CNNLSTMBaseline(): "cnn_lstm_baseline",
        FlowEnhancedLNN(): "LNN_model",
    }

    for model in model_dict:
        print(f"\nEvaluating {model_dict[model]}...")
        model.load_state_dict(
            torch.load(f"{model_dict[model]}.pth", map_location=device)
        )
        model.to(device)
        evaluate(
            model, dataloader, device, save_dir=f"eval_visuals_{model_dict[model]}"
        )
        print()


if __name__ == "__main__":
    main()
