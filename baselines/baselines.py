class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        base_model = efficientnet_b0(pretrained=True)
        self.encoder = nn.Sequential(*list(base_model.features.children()))
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1280, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid())

    def forward(self, x):  # x: [B, 3, H, W]
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
class CNNLSTM(nn.Module):
    def __init__(self, hidden_dim=512, lstm_layers=1):
        super().__init__()
        base_model = efficientnet_b0(pretrained=True)
        self.encoder = nn.Sequential(*list(base_model.features.children()))
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.lstm = nn.LSTM(input_size=1280, hidden_size=hidden_dim, num_layers=lstm_layers, batch_first=True)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 512 * 10 * 10),
            nn.ReLU(inplace=True),
            nn.Unflatten(1, (512, 10, 10)),
            nn.ConvTranspose2d(512, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid())

    def forward(self, x_seq):  
        batch_size, seq_len, _, H, W = x_seq.size()
        features = []

        for t in range(seq_len):
            f = self.encoder(x_seq[:, t])
            f = self.global_pool(f).squeeze(-1).squeeze(-1)  
            features.append(f)

        feature_seq = torch.stack(features, dim=1) 
        lstm_out, _ = self.lstm(feature_seq)
        last_output = lstm_out[:, -1, :] 

        out = self.decoder(last_output)
        return out 
    
def get_baseline_model(name, device='cuda'):
    if name == 'cnn_only':
        model = CNN()
    elif name == 'cnn_lstm':
        model = CNNLSTM()
    else:
        raise ValueError(f"Unknown model name: {name}")
    return model.to(device)

def train_baseline(model, dataloader, optimizer, device):
    model.train()
    metrics = defaultdict(float)
    count = 0

    for batch in tqdm(dataloader, desc="Training"):
        frames = batch['rgb'].to(device)
        target = batch['target'].to(device)

        optimizer.zero_grad()
        if hasattr(model, 'lstm'):
            pred_frame = model(frames)
        else:
            pred_frame = model(frames[:, -1])

        loss_recon = F.mse_loss(pred_frame, target)
        ssim_val = ssim(pred_frame, target, data_range=1.0, size_average=True)
        psnr_val = psnr(pred_frame, target)

        total_loss = loss_recon + (1 - ssim_val)
        total_loss.backward()
        optimizer.step()

        metrics['total'] += total_loss.item()
        metrics['recon'] += loss_recon.item()
        metrics['psnr'] += psnr_val.item()
        metrics['ssim'] += ssim_val.item()
        count += 1

    for k in metrics:
        metrics[k] /= count
    return metrics



