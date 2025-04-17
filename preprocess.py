import subprocess
import json
import os
import os
from PIL import Image
import random
from torchvision import transforms
import torch
from tqdm import tqdm


def get_rotation(video_path):
    cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
           '-show_entries', 'stream_tags=rotate', '-of', 'json', video_path]
    result = subprocess.run(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True)
    output = json.loads(result.stdout)
    rotate = output.get("streams", [{}])[0].get("tags", {}).get("rotate", "0")
    return int(rotate)


def extract_frames(video_path, output_dir, rotation, fps=None):
    os.makedirs(output_dir, exist_ok=True)

    transpose_map = {
        90: 'transpose=1',        # 90 clockwise
        180: 'transpose=2,transpose=2',  # 180 degrees = 2x 90ccw
        270: 'transpose=2'}       # 90 counter-clockwise

    filters = transpose_map.get(rotation, None)

    # Build command
    cmd = ["ffmpeg", "-i", video_path, "-map_metadata", "-1"]

    vf_filters = []
    if filters:
        vf_filters.append(filters)
    if fps is not None:
        vf_filters.append(f"fps={fps}")
    if vf_filters:
        cmd += ["-vf", ",".join(vf_filters)]

    cmd += [os.path.join(output_dir, "frame_%04d.jpg")]

    subprocess.run(cmd)


video_folder = "./train"
output_root = "./ffmpegframes"

for video_file in os.listdir(video_folder):
    if video_file.endswith(('.mp4', '.mov', '.avi')):
        video_path = os.path.join(video_folder, video_file)
        rotation = get_rotation(video_path)
        print(f"Processing {video_file} (rotation: {rotation})")
        output_dir = os.path.join(output_root, os.path.splitext(video_file)[0])
        extract_frames(video_path, output_dir, rotation, fps=5)


random.seed(42)

transform = transforms.Compose(
    [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])

main_folder = './ffmpegframes'
output_root = './transformedframes'  # Save processed frames here
splits = ['train', 'val', 'test']
os.makedirs(output_root, exist_ok=True)

all_folders = sorted([f for f in os.listdir(main_folder)
                     if os.path.isdir(os.path.join(main_folder, f))])

# Shuffle and split
random.shuffle(all_folders)
num_total = len(all_folders)
num_train = int(0.7 * num_total)
num_val = int(0.2 * num_total)

train_folders = all_folders[:num_train]
val_folders = all_folders[num_train:num_train + num_val]
test_folders = all_folders[num_train + num_val:]

split_map = {
    'train': train_folders,
    'val': val_folders,
    'test': test_folders}

for split in splits:
    print(f"Processing {split} set with {len(split_map[split])} folders...")
    for video_folder in tqdm(split_map[split]):
        video_path = os.path.join(main_folder, video_folder)
        output_video_path = os.path.join(output_root, split, video_folder)
        os.makedirs(output_video_path, exist_ok=True)

        for frame_file in sorted(os.listdir(video_path)):
            if not frame_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            input_frame_path = os.path.join(video_path, frame_file)
            output_frame_path = os.path.join(
                output_video_path, frame_file.replace('.jpg', '.pt'))

            img = Image.open(input_frame_path).convert('RGB')
            transformed = transform(img)

            torch.save(transformed, output_frame_path)
