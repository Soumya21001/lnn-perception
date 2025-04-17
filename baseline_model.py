import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from tqdm import tqdm
from core.lnn_model import FeatureExtractor

class CNNBaseline(nn.Module):
    def __init__(self, feature_dim=128):
        super().__init__()
        self.feature_extractor = FeatureExtractor()
        self.decoder = nn.Linear(feature_dim, 3 * 32 * 32)
        self.decoder_bn = nn.BatchNorm1d(3 * 32 * 32)
        self.flow_head = nn.Sequential(
            nn.Linear(feature_dim, 32 * 4 * 4),
            nn.ReLU(),
            nn.Unflatten(1, (32, 4, 4)),
            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 4, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(4, 2, 4, 2, 1),
            nn.Tanh())
    
    def forward(self, frames):
        B, T, C, H, W = frames.shape
        features = [self.feature_extractor(frames[:, t]) for t in range(T)]  
        features = torch.stack(features, dim=1) 

        pred_frame_feat = features[:, -1]  # [B, D]
        pred_frame = self.decoder(pred_frame_feat)
        pred_frame = self.decoder_bn(pred_frame).view(B, 3, 32 , 32 ).clamp(0, 1)
        pred_frame = F.interpolate(pred_frame,size=(64,64),mode='bilinear',align_corners=False)

        pred_flows = []
        for t in range(1, T):
            motion_feat = features[:, t] - features[:, t - 1].detach()
            x = self.flow_head[0](motion_feat)         
            x = self.flow_head[1](x)                   
            x = self.flow_head[2](x).view(B, 32, 4, 4)  
            flow = self.flow_head[3:](x)                
            flow = F.interpolate(flow, size=(224, 224), mode="bilinear", align_corners=True)
            pred_flows.append(flow)

        pred_flows = torch.stack(pred_flows, dim=1) 

        return pred_frame, pred_flows


class CNNLSTMBaseline(nn.Module):
    def __init__(self, feature_dim=128, hidden_dim=128):
        super().__init__()
        self.feature_extractor = FeatureExtractor()
        self.lstm = nn.LSTM(feature_dim, hidden_dim, batch_first=True)
        self.decoder = nn.Linear(hidden_dim, 3 * 32 * 32)
        self.decoder_bn = nn.BatchNorm1d(3 * 32 * 32)
        self.flow_head = nn.Sequential(
            nn.Linear(hidden_dim, 32 * 4 * 4),
            nn.ReLU(),
            nn.Unflatten(1, (32, 4, 4)),
            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 4, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(4, 2, 4, 2, 1),
            nn.Tanh())

    def forward(self, frames):
        B, T, C, H, W = frames.shape
        features = [self.feature_extractor(frames[:, t]) for t in range(T)]
        features = torch.stack(features, dim=1)
        lstm_out, _ = self.lstm(features)
        h_t = lstm_out[:, -1]
        pred_frame = self.decoder(h_t)
        pred_frame = self.decoder_bn(pred_frame).view(B, 3, 32, 32).clamp(0, 1)
        pred_frame = F.interpolate(pred_frame,size=(64,64),mode='bilinear',align_corners=False)
        
        pred_flows = []
        for t in range(1, T):
            motion_feat = lstm_out[:, t] - lstm_out[:, t - 1].detach() 
            x = self.flow_head[0](motion_feat)  
            x = self.flow_head[1](x)            
            x = self.flow_head[2](x).view(B, 32, 4, 4)  
            flow = self.flow_head[3:](x)        
            flow = F.interpolate(flow, size=(224,224), mode="bilinear", align_corners=True)
            pred_flows.append(flow)

        pred_flows = torch.stack(pred_flows, dim=1)  
        return pred_frame, pred_flows



