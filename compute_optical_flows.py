import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
#  Import the RAFT model and utility padder from the official repo
from raft import RAFT
from utils.utils import InputPadder

import torch
torch.backends.mps.is_available()

def load_model(model_path='models/raft-kitti.pth'):
    from argparse import Namespace
    args = Namespace(small=False, mixed_precision=False, alternate_corr=False)
    model = RAFT(args)

    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)

    model = model.eval().to('mps')
    return model


def load_frame(path):
    tensor = torch.load(path, map_location=torch.device('mps'))
    return tensor.unsqueeze(0)  # add batch dim → [1, 3, H, W]


def save_flow(flow_tensor, path):
    flow_np = flow_tensor[0].permute(1, 2, 0).detach().cpu().numpy()  # [H, W, 2]
    np.save(path, flow_np)


def generate_flow_for_dataset(input_root, output_root, model):
    splits = ['train', 'val', 'test']
    for split in splits:
        print(f"\n🔁 Processing {split} set...")
        input_split = os.path.join(input_root, split)
        output_split = os.path.join(output_root, split)
        os.makedirs(output_split, exist_ok=True)

        for video_folder in tqdm(sorted(os.listdir(input_split))):
            input_video_dir = os.path.join(input_split, video_folder)
            output_video_dir = os.path.join(output_split, video_folder)
            os.makedirs(output_video_dir, exist_ok=True)

            frames = sorted(f for f in os.listdir(input_video_dir) if f.endswith('.pt'))
            for i in range(len(frames) - 1):
                frame1_path = os.path.join(input_video_dir, frames[i])
                frame2_path = os.path.join(input_video_dir, frames[i+1])
                output_path = os.path.join(output_video_dir, f"flow_{i:04d}.npy")

                image1 = load_frame(frame1_path)
                image2 = load_frame(frame2_path)
                padder = InputPadder(image1.shape)
                image1, image2 = padder.pad(image1, image2)

                with torch.no_grad():
                    _, flow_up = model(image1, image2, iters=20, test_mode=True)

                save_flow(flow_up, output_path)


if __name__ == '__main__':
    frames_root = '/Users/st/Projects/BDD100K/transformed_frames'
    flow_output_root = '/Users/st/Projects/BDD100K/optical_flow'
    model = load_model('models/raft-kitti.pth')

    generate_flow_for_dataset(frames_root, flow_output_root, model)

