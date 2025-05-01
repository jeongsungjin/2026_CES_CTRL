import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('TkAgg')  # <-- 중요

import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import random
import os

# ==========================
# ConvLSTM Cell
# ==========================
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(input_dim + hidden_dim, hidden_dim * 4, kernel_size, padding=padding)

    def forward(self, x, hidden):
        h, c = hidden
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size, spatial_size):
        height, width = spatial_size
        h = torch.zeros(batch_size, 64, height, width)
        c = torch.zeros(batch_size, 64, height, width)
        return h.to(next(self.parameters()).device), c.to(next(self.parameters()).device)

# ==========================
# ConvLSTM Model
# ==========================
class ConvLSTM(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, kernel_size=3, pred_frames=5):
        super().__init__()
        self.encoder = ConvLSTMCell(input_dim, hidden_dim, kernel_size)
        self.decoder = ConvLSTMCell(input_dim, hidden_dim, kernel_size)
        self.conv_out = nn.Conv2d(hidden_dim, input_dim, kernel_size=1)
        self.pred_frames = pred_frames

    def forward(self, input_tensor):
        b, seq_len, c, h, w = input_tensor.size()
        h_t, c_t = self.encoder.init_hidden(b, (h, w))

        # Encoding
        for t in range(seq_len):
            h_t, c_t = self.encoder(input_tensor[:, t], (h_t, c_t))

        # Decoding
        outputs = []
        decoder_input = input_tensor[:, -1]
        for _ in range(self.pred_frames):
            h_t, c_t = self.decoder(decoder_input, (h_t, c_t))
            out = self.conv_out(h_t)
            outputs.append(out)
            decoder_input = out  # next input
        return torch.stack(outputs, dim=1)

# ==========================
# Dataset
# ==========================
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

# ==========================
# Training
# ==========================
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

# ==========================
# Visualization
# ==========================
def visualize_prediction(x_seq, y_target, y_pred, epoch):
    pred_frames = y_pred.shape[0]
    fig, axs = plt.subplots(2, pred_frames, figsize=(4 * pred_frames, 8))

    for i in range(pred_frames):
        axs[0, i].imshow(y_target[i, 0], cmap='gray')
        axs[0, i].set_title(f"GT Frame {i+1}")
        axs[0, i].axis('off')

        axs[1, i].imshow(torch.sigmoid(torch.tensor(y_pred[i, 0])).numpy(), cmap='gray')
        axs[1, i].set_title(f"Pred Frame {i+1}")
        axs[1, i].axis('off')

    plt.tight_layout()
    os.makedirs("visuals", exist_ok=True)
    plt.savefig(f"visuals/prediction_epoch_{epoch}.png")
    plt.close()

# ==========================
# Main
# ==========================
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

    model = ConvLSTM(pred_frames=pred_frames).to(device)
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
