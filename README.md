# Adaptive Vision-Based Perception & Navigation for Autonomous Vehicles

**Author:** Soumya Taneja  
**Institute:** Rochester Institute of Technology (MS in Artificial Intelligence)  
**Project Type:** Capstone Research Project  
**Semester:** Spring 2025

## Overview

This project presents **FlowEnhancedLNN**, a novel architecture that combines a Liquid Time-Constant Neural Network (LNN) with an image decoder and optical flow module for spatiotemporal video prediction. It is benchmarked against CNN-only and CNN+LSTM baselines. The model is trained on a preprocessed version of the BDD100K driving dataset, aiming to predict the next video frame and optical flow simultaneously.

## Project Figure


```plaintext
           ┌────────────────────────┐
           │ 1) Raw Video Frames    |
           │    X = [x₁, x₂, …, xₜ]  | 
           └─────────────┬──────────┘
                         │
                         ▼
           ┌────────────────────────┐
           │ 2) Preprocessing       │
           │    • Resize → .pt      │
           └─────────────┬──────────┘
                         │
                         ▼
           ┌────────────────────────┐
           │ 3) Transformed Frames  │
           │    {train,val,test}/…  │
           └─────────────┬──────────┘
                         │
                         ▼
           ┌────────────────────────┐
           │ 4) RAFT Optical Flow   │
           │    • Compute F̂ₜ → .npy  │
           └─────────────┬──────────┘
                         │
                         ▼
           ┌────────────────────────┐
           │ 5) Feature Extraction  │
           │    CNN → f₁, f₂, …, fₜ  │
           └─────────────┬──────────┘
                         │
                         ▼
           ┌────────────────────────┐
           │ 6) LNN Temporal Module │
           │    Update hidden hₜ     │
           └───────┬───────┬────────┘
                   │       │
           ┌───────▼───┐   │
           │ Frame     │   │
           │ Decoder   │   │
           │ (Linear + │   │
           │ Upsample) │   │
           └───────┬───┘   │
                   │       │
                   │       │
                   │       ▼
                   │   ┌───────────────┐
                   │   │ Flow Decoder  │
                   │   │(ConvTranspose)│
                   │   └──────┬────────┘
                   │          │
                   │          │
                   ▼          ▼
           ┌───────────────────┐
           │ 7) Predictions    |  
           │  • Frame ŷₜ        |    
           │  • Flow F̂ₜ         |    
           └─────────────┬─────┘
                         │
                         ▼
           ┌────────────────────────────┐
           │ 8) Loss Module             │
           │  • ℓ_recon = MSE(ŷₜ,xₜ)     │  
           │  • ℓ_ssim  = 1–SSIM(…)     │
           │  • ℓ_psnr  = PSNR(…)       │
           │  • ℓ_contr = Contrastive   │
           │  • ℓ_flow  = L₁(F̂ₜ,Fₜ)      │
           │  • ℓ_smooth = Smoothness   │
           │  • ℓ_total = α(…)+β(…)+γ(…)│
           └─────────────┬──────────────┘
                         │
                         ▼
           ┌────────────────────────┐
           │ 9) Backprop & Update   │
           │    via Adam Optimizer  │
           └─────────────┬──────────┘
                         │
                         ▼
           ┌────────────────────────┐
           │10) Evaluation          │
           │    • SSIM, PSNR, …     │
           │    • Save metrics      │
           │    • Visualize results │
           └────────────────────────┘
```

## Getting Started

## Prerequisites
	• Linux or macOS machine (GPU recommended)
	• Conda installed
	• Access to BDD100K dataset (link provided below)
	• RAFT repository cloned under core/raft/ 

### Environment Setup

# 1. Create & activate conda environment
```
conda create -n avmodel python=3.11 -y
conda activate avmodel
```

# 2. Install Python dependencies
```
pip install -r requirements.txt
```

# 3. Prepare RAFT code
```
git clone https://github.com/princeton-vl/RAFT core/raft
cd core/raft
pip install -r requirements.txt    # RAFT’s own deps
cd ../../
```



Note: The project is tested on CUDA-enabled GPUs and can also run on CPU with reduced performance.

# Dependencies
	•	torch, torchvision
	•	numpy, matplotlib, opencv-python
	•	pytorch_msssim
	•	tqdm, pandas, prefetch_generator
	•	RAFT model and utils: clone RAFT repo

# Directory Structure
```
├── core/
│   ├── raft/                  # RAFT repo clone & utils
│   |── utils/  	       # InputPadder, helper scripts
|   |-- models/		       # RAFT model paths
├── preprocess_frames.py       # Preprocess and save .pt frames
├── compute_optical_flow.py    # Generate .npy flows via RAFT
├── train.py                   # Train LNN model
├── train_baseline.py          # Train CNN and/or CNN+LSTM
├── evaluate.py                # Eval LNN, CNN, CNN-LSTM
├── lnn_model.py               # LNN architecture
├── baselines.py               # CNN & CNN-LSTM
├── datasetloader.py           # Custom DataLoader
├── loss_functions.py          # PSNR, SSIM, Flow Loss, Reconstruction Loss, etc.
└── visualisations/            # Prediction & flow visualizations
    ├── eval_visuals_LNN_model/
    ├── eval_visuals_cnn_baseline/
    └── eval_visuals_cnn_lstm_baseline/                  
├── README.md  
└── requirements.txt
```

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
clone RAFT repo under core/
```
git clone https://github.com/princeton-vl/RAFT.git raft
```
Note: Adjust the python imports in your scripts 
 ```
   python compute_optical_flow.py \
  --model-path raft/models/raft-kitti.pth \
  --input-root transformed_frames \
  --output-root optical_flow
  ```
Internally this script:
	1. Loads RAFT with raft-kitti.pth.
	2. Pads and runs pairwise frames.
	3. Saves flow_XXXX.npy under optical_flow/{split}/{video_id}/.

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
		•Rubanova et al., **“Latent Ordinary Differential Equations for Irregularly‑Sampled Time Series”**, NeurIPS 2019.  
                •Pan et al., **“Liquid Time‑Constant Networks for Continuous‑Time Sequence Modeling”**, ICML Workshop 2022.  
	•Special Thanks: RIT AI faculty and Prof Zhiqiang Tao



