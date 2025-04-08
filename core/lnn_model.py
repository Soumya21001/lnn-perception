import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class FeatureExtractor(nn.Module):
    def __init__(self, input_channels=3, output_dim=64):
        super().__init__()
        
        # Load pretrained ResNet18 and modify
        resnet = models.resnet18(pretrained=True)
        
        if input_channels != 3:
            resnet.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool)
        
        self.project = nn.Linear(512, output_dim)

    def forward(self, x):
        x = self.backbone(x)         # [B, 512, 1, 1]
        x = x.view(x.size(0), -1)    # Flatten to [B, 512]
        x = self.project(x)          # Project to [B, 64]
        return x
    
class LTCCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.time_constants = nn.Parameter(torch.rand(hidden_dim) * 0.9 + 0.1)
        self.W_input = nn.Linear(input_dim, hidden_dim)
        self.W_recurrent = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.leak = nn.Parameter(torch.tensor(0.1))

    def forward(self, x, h_prev=None):
        if h_prev is None:
            h_prev = torch.zeros(x.shape[0], self.hidden_dim).to(x.device)
        combined = self.W_input(x) + self.W_recurrent(h_prev)
        gates = torch.sigmoid(combined)
        effective_time = self.time_constants * gates
        dh_dt = (-h_prev + torch.tanh(combined)) / (effective_time + 1e-6)
        h_new = h_prev + self.leak * dh_dt
        return h_new


class FlowEnhancedLNN(nn.Module):
    def __init__(self, feature_dim=64, hidden_dim=64):
        super().__init__()
        self.feature_extractor = FeatureExtractor()
        self.lnn_cell = LTCCell(feature_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, 3 * 32 * 32)
        self.decoder_bn = nn.BatchNorm1d(3 * 32 * 32)
        
        self.flow_head = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 4, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(4, 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(2, 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(2, 2, kernel_size=2, stride=2, padding=1, output_padding=0),
            nn.Tanh()
        )
        self.flow_embed = nn.Linear(hidden_dim, 32 * 4 * 4)
    
    def forward(self, frames):
        batch_size, seq_len, c, h, w = frames.size()
        h_states = []
        h_prev = None
        for t in range(seq_len):
            features = self.feature_extractor(frames[:, t])
            h_t = self.lnn_cell(features, h_prev)
            h_states.append(h_t)
            h_prev = h_t

        pred_frame = self.decoder(h_states[-1])  # [B, 3*32*32]
        pred_frame = self.decoder_bn(pred_frame)  # Align distribution with normalized targets
        pred_frame = pred_frame.view(batch_size, 3, 32, 32)

        pred_flows = []
        for t in range(1, seq_len):
            motion_feat = h_states[t] - h_states[t - 1].detach()
            x = self.flow_embed(motion_feat).view(batch_size, 32, 4, 4)
            flow = self.flow_head(x)
            flow = F.interpolate(flow, size=(224, 224), mode='bilinear', align_corners=True)
            pred_flows.append(flow)
        pred_flows = torch.stack(pred_flows, dim=1)  # [B, seq_len-1, 2, 224, 224]

        return pred_frame, pred_flows

