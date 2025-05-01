import math
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import random
import os

# --------------------------
# Positional Encoding Module
# --------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x: (seq_len, batch, d_model)
        x = x + self.pe[:, :x.size(0)].transpose(0,1)
        return x

# --------------------------
# CNN Encoder: extracts latent vector from image
# --------------------------
class CNNEncoder(nn.Module):
    def __init__(self, in_channels=1, hidden_dim=64):
        super(CNNEncoder, self).__init__()
        # 128x128 input -> (B, hidden_dim, 16, 16) after 3 conv layers with stride 2
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1)   # -> (B,32,64,64)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)            # -> (B,64,32,32)
        self.conv3 = nn.Conv2d(64, hidden_dim, kernel_size=3, stride=2, padding=1)      # -> (B,hidden_dim,16,16)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # x: (batch, channels, H, W)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        # Flatten spatial dimensions: output shape (batch, hidden_dim * 16 * 16)
        x = x.view(x.size(0), -1)
        return x

# --------------------------
# CNN Decoder: reconstruct image from latent vector
# --------------------------
class CNNDecoder(nn.Module):
    def __init__(self, out_channels=1, hidden_dim=64):
        super(CNNDecoder, self).__init__()
        # Input latent feature will be reshaped to (batch, hidden_dim, 16, 16)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # -> (B,64,32,32)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),          # -> (B,32,64,64)
            nn.ReLU(),
            nn.ConvTranspose2d(32, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)   # -> (B,out_channels,128,128)
        )
        
    def forward(self, x):
        # x: (batch, hidden_dim, 16, 16)
        return self.decoder(x)

# --------------------------
# CNN + Transformer Model for Occupancy Grid Prediction
# --------------------------
class CNNTransformer(nn.Module):
    def __init__(self, in_channels=1, hidden_dim=64, d_model=512, nhead=8, num_encoder_layers=3,
                 num_decoder_layers=3, pred_frames=5):
        super(CNNTransformer, self).__init__()
        self.pred_frames = pred_frames
        self.hidden_dim = hidden_dim
        self.encoder_cnn = CNNEncoder(in_channels=in_channels, hidden_dim=hidden_dim)
        # CNNEncoder output dimension: hidden_dim * 16 * 16
        self.cnn_feature_dim = hidden_dim * 16 * 16
        # Project CNN features to Transformer dimension (d_model)
        self.proj = nn.Linear(self.cnn_feature_dim, d_model)
        self.d_model = d_model
        
        # Positional Encoding modules for encoder and decoder
        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=500)
        self.pos_decoder = PositionalEncoding(d_model=d_model, max_len=500)
        
        # Transformer Encoder: processes input sequence
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        # Transformer Decoder: generates future latent sequence tokens
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dropout=0.1)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        
        # Learnable tokens for decoder input: shape (pred_frames, d_model)
        self.decoder_input = nn.Parameter(torch.zeros(pred_frames, d_model))
        
        # Map decoder output tokens back to CNN latent space
        self.fc = nn.Linear(d_model, self.cnn_feature_dim)
        self.decoder_cnn = CNNDecoder(out_channels=in_channels, hidden_dim=hidden_dim)
        
    def forward(self, x):
        # x: (batch, seq_len, channels, H, W) where H=W=128 expected
        b, seq_len, C, H, W = x.size()
        latent_seq = []
        for t in range(seq_len):
            frame = x[:, t]  # (batch, C, H, W)
            feat = self.encoder_cnn(frame)  # (batch, cnn_feature_dim)
            latent = self.proj(feat)        # (batch, d_model)
            latent_seq.append(latent)
        latent_seq = torch.stack(latent_seq, dim=1)  # (batch, seq_len, d_model)
        
        # Transformer expects input shape: (seq_len, batch, d_model)
        encoder_input = latent_seq.transpose(0, 1)  # (seq_len, batch, d_model)
        encoder_input = self.pos_encoder(encoder_input)
        memory = self.transformer_encoder(encoder_input)  # (seq_len, batch, d_model)
        
        # Prepare decoder input tokens: use learnable tokens, expand for batch
        decoder_input = self.decoder_input.unsqueeze(1).repeat(1, b, 1)  # (pred_frames, batch, d_model)
        decoder_input = self.pos_decoder(decoder_input)
        # Transformer decoder: outputs predicted latent tokens for future frames
        decoder_output = self.transformer_decoder(tgt=decoder_input, memory=memory)  # (pred_frames, batch, d_model)
        
        # Transpose back to (batch, pred_frames, d_model)
        decoder_output = decoder_output.transpose(0, 1)  # (batch, pred_frames, d_model)
        # Map each token to CNN latent space
        latent_pred = self.fc(decoder_output)  # (batch, pred_frames, cnn_feature_dim)
        # Reshape to image feature maps: (batch * pred_frames, hidden_dim, 16, 16)
        latent_pred = latent_pred.view(b * self.pred_frames, self.hidden_dim, 16, 16)
        # Reconstruct images using CNN decoder
        out = self.decoder_cnn(latent_pred)  # (b * pred_frames, C, 128, 128)
        # Reshape to (batch, pred_frames, C, 128, 128)
        out = out.view(b, self.pred_frames, C, 128, 128)
        return out

# --------------------------
# Dataset (OccupancyGridDataset)
# --------------------------
class OccupancyGridDataset(Dataset):
    def __init__(self, npy_file, input_frames=10, pred_frames=5, step=1, augment=False):
        self.raw_data = np.load(npy_file)
        self.input_frames = input_frames
        self.pred_frames = pred_frames
        self.augment = augment
        self.data = []
        for i in range(0, self.raw_data.shape[0] - input_frames - pred_frames + 1, step):
            x_seq = self.raw_data[i:i+input_frames]
            y_seq = self.raw_data[i+input_frames:i+input_frames+pred_frames]
            self.data.append((x_seq, y_seq))
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        x, y = self.data[idx]
        x = torch.tensor(x[:, None, :, :], dtype=torch.float32)
        y = torch.tensor(y[:, None, :, :], dtype=torch.float32)
        if self.augment:
            if random.random() < 0.5:
                x = TF.hflip(x)
                y = TF.hflip(y)
            if random.random() < 0.3:
                noise = torch.randn_like(x) * 0.05
                x = torch.clamp(x + noise, 0, 1)
        return x, y

# --------------------------
# Training and Evaluation
# --------------------------
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            total_loss += loss.item()
    return total_loss / len(dataloader)

# --------------------------
# Visualization
# --------------------------
def visualize_prediction(x_seq, y_target, y_pred, epoch):
    pred_frames = y_pred.shape[0]
    fig, axs = plt.subplots(2, pred_frames, figsize=(4 * pred_frames, 8))
    for i in range(pred_frames):
        axs[0, i].imshow(y_target[i, 0], cmap='gray', vmin=0, vmax=1)
        axs[0, i].set_title(f"GT Frame {i+1}")
        axs[0, i].axis('off')
        # Apply sigmoid to prediction for visualization
        axs[1, i].imshow(torch.sigmoid(torch.tensor(y_pred[i, 0])).numpy(), cmap='gray', vmin=0, vmax=1)
        axs[1, i].set_title(f"Pred Frame {i+1}")
        axs[1, i].axis('off')
    plt.tight_layout()
    os.makedirs("visuals", exist_ok=True)
    plt.savefig(f"visuals/prediction_epoch_{epoch}.png")
    plt.close()

# --------------------------
# Main Function
# --------------------------
def main():
    npy_file = "vehicle_like_sequence_300.npy"
    input_frames = 10
    pred_frames = 5
    batch_size = 8
    epochs = 30
    lr = 1e-3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset = OccupancyGridDataset(npy_file, input_frames=input_frames, pred_frames=pred_frames, step=1, augment=True)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    
    model = CNNTransformer(in_channels=1, hidden_dim=64, d_model=512, nhead=8,
                           num_encoder_layers=3, num_decoder_layers=3, pred_frames=pred_frames).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    
    for epoch in range(epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        print(f"[{epoch+1:02d}/{epochs}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        if (epoch + 1) % 1 == 0:
            model.eval()
            with torch.no_grad():
                x_seq, y_target = dataset[0]
                x_seq = x_seq.unsqueeze(0).to(device)
                y_pred = model(x_seq).squeeze(0).cpu().numpy()
                visualize_prediction(x_seq[0].cpu().numpy(), y_target.numpy(), y_pred, epoch + 1)
                
if __name__ == "__main__":
    main()
