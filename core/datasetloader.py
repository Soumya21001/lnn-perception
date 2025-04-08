import os
import re
from glob import glob
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from prefetch_generator import BackgroundGenerator

class FastDataLoader(DataLoader):
    """A DataLoader wrapper that prefetches data in the background."""
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

class AVPerceptionDataset(Dataset):
    def __init__(self, frame_root, flow_root, split='train', seq_length=5, transform=None, cache_data=False):
        self.seq_length = seq_length
        self.transform = transform
        self.cache_data = cache_data
        self.frame_cache = {} if cache_data else None
        self.flow_cache = {} if cache_data else None
        self.samples = []

        frame_split_dir = os.path.join(frame_root, split)
        flow_split_dir = os.path.join(flow_root, split)

        video_ids = sorted(os.listdir(frame_split_dir))
        print(f"Found {len(video_ids)} video folders in split '{split}'")

        for vid in video_ids:
            frame_dir = os.path.join(frame_split_dir, vid)
            flow_dir = os.path.join(flow_split_dir, vid)

            if not os.path.isdir(frame_dir) or not os.path.isdir(flow_dir):
                continue

            frame_paths = sorted(glob(os.path.join(frame_dir, "frame_*.pt")), key=self._frame_sort_key)
            flow_paths = sorted(glob(os.path.join(flow_dir, "flow_*.npy")), key=self._flow_sort_key)

            if len(frame_paths) < seq_length or len(flow_paths) < seq_length - 1:
                print(f"Skipping video {vid}: Not enough data ({len(frame_paths)} frames, {len(flow_paths)} flows)")
                continue

            for i in range(len(frame_paths) - seq_length + 1):
                seq_frames = frame_paths[i:i + seq_length]
                seq_flows = flow_paths[i:i + seq_length - 1]
                self.samples.append((seq_frames, seq_flows))

        print(f"Total valid sequences collected: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame_paths, flow_paths = self.samples[idx]
        

        frames = []
        for fp in frame_paths:
            if self.cache_data and fp in self.frame_cache:
                frame = self.frame_cache[fp]
            else:
                frame = torch.load(fp)
                if self.cache_data:
                    self.frame_cache[fp] = frame
            if self.transform:
                frame = self.transform(frame)
            frames.append(frame)
        frames = torch.stack(frames)  # (T, 3, H, W)

        flows = []
        for fp in flow_paths:
            if self.cache_data and fp in self.flow_cache:
                flow = self.flow_cache[fp]
            else:
                flow_np = np.load(fp)  # shape: (H, W, 2)
    
                flow_np = cv2.resize(flow_np, (224, 224), interpolation=cv2.INTER_LINEAR)
                flow = torch.from_numpy(flow_np).permute(2, 0, 1).float().div(224.0)
                if self.cache_data:
                    self.flow_cache[fp] = flow
            flows.append(flow)
        flows = torch.stack(flows)  # (T-1, 2, H, W)

        return frames, flows

    def _frame_sort_key(self, path):
        match = re.search(r"frame_(\d+)\.pt", path)
        return int(match.group(1)) if match else -1

    def _flow_sort_key(self, path):
        match = re.search(r"flow_(\d+)\.npy", path)
        return int(match.group(1)) if match else -1