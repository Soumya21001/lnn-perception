import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import ToPILImage, Resize, ToTensor
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
import os
from core.datasetloader import AVPerceptionDataset, FastDataLoader
from core.loss_functions import psnr_score, ssim_score, temporal_contrastive_loss, optical_flow_consistency_loss, smooth_loss
from baselines import CNNBaseline, CNNLSTMBaseline

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

            frames, flow_gt = frames.to(device), flow_gt.to(device)
            target = frames[:, -1].clamp(0, 1)

            pred_frame, pred_flow = model(frames)
            pred_frame = pred_frame.clamp(0, 1)
            pred_flow = pred_flow.clamp(0, 1)

            loss_recon = F.mse_loss(pred_frame, target)
            psnr = psnr_score(pred_frame, target)
            ssim = ssim_score(pred_frame, target)

            features = [model.feature_extractor(frames[:, t]) for t in range(frames.size(1))]
            features = torch.stack(features, dim=1)

            contrastive = temporal_contrastive_loss(features)
            flow_loss = optical_flow_consistency_loss(pred_flow, flow_gt)
            smooth = smooth_loss(pred_flow)

            metrics['recon'] += loss_recon.item()
            metrics['psnr'] += psnr.item()
            metrics['ssim'] += ssim.item()
            metrics['contrastive'] += contrastive.item()
            metrics['flow'] += flow_loss.item()
            metrics['smooth'] += smooth.item()
            count += 1

            # Save one visualization
            if batch_idx == 0:
                visualize_prediction(pred_frame, target, os.path.join(save_dir, "./sample_eval.png"))

    for k in metrics:
        metrics[k] /= count

    print("\n Evaluation Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")


# ---------- Main ----------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([ToPILImage(), Resize(64), ToTensor()])

    dataset = AVPerceptionDataset("transformed_frames/", "optical_flow/", split='val', seq_length=5, transform=transform)
    dataloader = FastDataLoader(dataset, batch_size=64, shuffle=False, num_workers=6)

    baseline_type = input("Evaluate which baseline? (cnn/cnn_lstm): ").strip().lower()

    if baseline_type == "cnn":
        model = CNNBaseline()
        model.load_state_dict(torch.load("cnn_baseline.pth", map_location=device))
    elif baseline_type == "cnn_lstm":
        model = CNNLSTMBaseline()
        model.load_state_dict(torch.load("cnn_lstm_baseline.pth", map_location=device))
    else:
        raise ValueError("Invalid model name")

    model.to(device)
    evaluate(model, dataloader, device, save_dir= f"./ {baseline_type}_eval_visuals")


if __name__ == "__main__":
    main()