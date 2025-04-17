# Adaptive Vision-Based Perception & Navigation for Autonomous Vehicles

**Author:** Soumya Taneja  
**Institute:** Rochester Institute of Technology (MS in Artificial Intelligence)  
**Project Type:** Capstone Research Project  
**Semester:** Spring 2025

## Overview

This project presents **FlowEnhancedLNN**, a novel architecture that combines a Liquid Time-Constant Neural Network (LNN) with an image decoder and optical flow module for spatiotemporal video prediction. It is benchmarked against CNN-only and CNN+LSTM baselines. The model is trained on a preprocessed version of the BDD100K driving dataset, aiming to predict the next video frame and optical flow simultaneously.

## Project Figure

## Getting Started

### Environment Setup

```bash
conda create -n avmodel python=3.11 -y
conda activate avmodel
pip install -r requirements.txt
```


Note: The project is tested on CUDA-enabled GPUs and can also run on CPU with reduced performance.

# Dependencies
	•	torch, torchvision
	•	numpy, matplotlib, opencv-python
	•	pytorch_msssim
	•	tqdm, pandas, prefetch_generator
	•	RAFT model and utils: clone RAFT repo

# Directory Structure
.
├── train.py                  # Train LNN
├── train_baseline.py         # Train baselines (CNN or CNN+LSTM)
├── evaluate.py               # Evaluate LNN model
├── evaluate_baseline.py      # Evaluate baselines
├── lnn_model.py              # LNN model architecture
├── baselines.py              # Baseline CNN and CNN+LSTM
├── datasetloader.py          # Custom Dataset + DataLoader
├── loss_functions.py         # All loss functions
├── visuals/                 # Saved frame predictions
├── checkpoints/             # Saved models
├── transformed_frames/      # Preprocessed frames (in .pt format)
├── optical_flow/            # Precomputed optical flow (.npy)
├── README.md
└── requirements.txt


# Data Preparation
1.	Download Raw BDD100K
Download from: https://www.kaggle.com/datasets/robikscube/driving-video-with-object-tracking

2.	Frame & Flow Preprocessing
Run preprocessing scripts
```bash
python preprocess_frames.py \
  --input ./videos/train \
  --output ./transformed_frames \
  --resize 224
```
Output:
```
transformed_frames/{train,val,test}/{video_id}/frame_0000.pt
```
3. Generate Optical Flow
 ```
   python compute_optical_flow.py \
  --model-path raft/models/raft-kitti.pth \
  --input-root transformed_frames \
  --output-root optical_flow
  ```

##  Key Results

| Model          | SSIM ↑ | PSNR ↑ | Flow Loss ↓ | Smoothness ↓ |
| -------------- | :----: | :----: | :---------: | :----------: |
| CNN Baseline   |  0.41  |  19.2  |    0.026    |    0.017     |
| CNN + LSTM     |  0.44  |  20.1  |    0.021    |    0.015     |
| **LNN (Proposed Model)** | **0.47** | **21.4** |  **0.017**  |  **0.013**   |


# References & Acknowledgements
	•RAFT: Optical flow backbone from princeton-vl/RAFT
	•BDD100K Dataset: bdd-data.berkeley.edu
	•LNN Theory:
		•Tallec & Ollivier, “Can Recurrent Neural Networks Warp Time?” (ICLR 2018)
		•Lai et al., “Modeling Receptive Fields with ODE Nets” (NeurIPS 2019)
	•Special Thanks: RIT AI faculty and Prof Zhiqiang Tao



