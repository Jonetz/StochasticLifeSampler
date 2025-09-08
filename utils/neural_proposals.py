from matplotlib import pyplot as plt

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

from utils.encodings import load_rle_files

import torch

def aug(envs, targets, device='cuda'):
    """
    envs: (B, 1, H, W)
    targets: (B, h, w)
    """
    B = envs.shape[0]
    envs = envs.to(device)
    targets = targets.to(device)

    # --- Random horizontal flip ---
    #hflip_mask = torch.rand(B, device=device) < 0.5
    #envs[hflip_mask] = torch.flip(envs[hflip_mask], dims=[3])
    #targets[hflip_mask] = torch.flip(targets[hflip_mask], dims=[2])

    # --- Random vertical flip ---
    #vflip_mask = torch.rand(B, device=device) < 0.5
    #envs[vflip_mask] = torch.flip(envs[vflip_mask], dims=[2])
    #targets[vflip_mask] = torch.flip(targets[vflip_mask], dims=[1])

    # --- Random 90° rotations ---
    #rotations = torch.randint(0, 4, (B,), device=device)  # 0,1,2,3
    #for k in range(1, 4):
    #    mask = rotations == k
    #    if mask.any():
    #        envs[mask] = torch.rot90(envs[mask], k, dims=[2,3])
    #        targets[mask] = torch.rot90(targets[mask], k, dims=[1,2])

    return envs, targets


class PatchNet(nn.Module):
    """
    Lightweight CNN to predict a fixed-size patch from an environment patch.
    Improved with more channels, center cropping instead of global average, and better receptive field.
    """
    def __init__(self, env_size=12, patch_size=5, dropout=0.15):
        super().__init__()
        self.env_size = env_size
        self.patch_size = patch_size

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, patch_size*patch_size, kernel_size=3, padding=1)
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x):
        # x: [B, 1, env_size, env_size]
        x = F.relu(self.conv1(x))
        x = self.dropout(x)
        x = F.relu(self.conv2(x))
        x = self.dropout(x)
        x = F.relu(self.conv3(x))
        x = self.dropout(x)
        x = self.conv4(x)  # [B, patch_size^2, H, W]

        # Crop the center patch region instead of global average
        center = self.env_size // 2
        half_patch = self.patch_size // 2
        x = x[:, :, center-half_patch:center+half_patch, center-half_patch:center+half_patch]

        # Flatten channels to match patch size
        x = x.mean(dim=[2,3])  # [B, patch_size^2] 
        x = torch.sigmoid(x)
        x = x.view(-1, self.patch_size, self.patch_size)  # [B, patch_size, patch_size]
        return x

class PatchDataset(Dataset):
    """
    Fast, sparse-aware dataset for (env, target) patches from binary arrays.
    Only generates windows around alive cells to avoid huge empty computations.
    """
    def __init__(self, arrs, env_size=12, patch_size=5, min_alive=2, device='cuda'):
        self.env_size = env_size
        self.patch_size = patch_size
        self.device = device

        env_list, target_list = [], []

        ci = env_size // 2 - patch_size // 2
        cj = env_size // 2 - patch_size // 2

        for arr in arrs:
            # Convert to torch tensor if needed
            if isinstance(arr, np.ndarray):
                arr = torch.from_numpy(arr)
            arr = arr.bool()

            # Skip empty arrays
            nonzero = arr.nonzero(as_tuple=False)
            if len(nonzero) == 0:
                continue

            # Trim zero borders
            top_left = nonzero.min(dim=0)[0]
            bottom_right = nonzero.max(dim=0)[0] + 1
            trimmed = arr[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]

            H, W = trimmed.shape
            if H < env_size or W < env_size:
                continue

            # PyTorch sliding window on trimmed array
            windows = trimmed.unfold(0, env_size, 1).unfold(1, env_size, 1)
            windows = windows.contiguous().view(-1, env_size, env_size)

            # Extract target patches
            targets = windows[:, ci:ci+patch_size, cj:cj+patch_size]

            # Filter by min_alive
            mask = targets.sum(dim=(1,2)) >= min_alive
            if mask.any():
                env_list.append(windows[mask])
                target_list.append(targets[mask])

        if len(env_list) == 0:
            # No samples
            self.envs = torch.empty(0, 1, env_size, env_size, dtype=torch.bool)
            self.targets = torch.empty(0, patch_size, patch_size, dtype=torch.bool)
            return

        # Concatenate surviving windows
        envs = torch.cat(env_list, dim=0)
        targets = torch.cat(target_list, dim=0)

        # Move to device once
        try:
            self.envs = envs.unsqueeze(1).to(device=device, dtype=torch.bool)
            self.targets = targets.to(device=device, dtype=torch.bool)
            print(f"Loaded {len(self.envs)} samples on {device}")
        except RuntimeError as e:
            print(f"GPU memory insufficient, keeping on CPU. ({e})")
            self.envs = envs.unsqueeze(1)
            self.targets = targets

    def __len__(self):
        return len(self.envs)

    def __getitem__(self, idx):
        # Convert to float32 only when feeding model
        env = self.envs[idx].float()
        target = self.targets[idx].float()
        return env, target

    
def weighted_bce(preds, targets, alpha=5.0):
    # preds, targets: [B, patch_size, patch_size]
    weight = torch.ones_like(targets)
    weight[targets == 1] = alpha
    loss = F.binary_cross_entropy(preds, targets, weight=weight)
    return loss

def visualize_training_progress(history):
    # Plot training history
    plt.figure(figsize=(6,4))
    plt.plot(range(1, len(history)+1), history, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Average BCE Loss')
    plt.title('PatchNet Training History')
    plt.grid(True)
    plt.show()

def train_patchnet(arrs, env_size=12, patch_size=5, min_alive=2, epochs=20, batch_size=64, lr=1e-3, device='cuda'):
    dataset = PatchDataset(arrs, env_size, patch_size, min_alive, device='cuda')

    net = PatchNet(env_size, patch_size, dropout=0.1).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    history = []  # store epoch losses
    
    # Split into train/val
    val_size = int(0.1 * len(dataset))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Model
    net = PatchNet(env_size, patch_size).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)

    # Training loop
    prev_loss = 1 
    print('Beginning Inference ...')
    for epoch in range(epochs):
        net.train()
        total_loss = 0
        for envs, targets in train_loader:
            envs, targets = envs.to(device), targets.to(device)
            #envs, targets = aug(envs, targets)
            optimizer.zero_grad()
            preds = net(envs)
            loss = weighted_bce(preds, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * envs.size(0)

        train_loss = total_loss / len(train_ds)
        history.append(train_loss)
        if total_loss < prev_loss:
            torch.save(net.state_dict(), r'data\network_intermediate.pth')   
        prev_loss = total_loss

        # Validation
        net.eval()
        val_loss = 0
        with torch.no_grad():
            for envs, targets in val_loader:
                envs, targets = envs.to(device), targets.to(device)
                preds = net(envs)
                loss = weighted_bce(preds, targets)
                val_loss += loss.item() * envs.size(0)
        val_loss /= len(val_ds)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    visualize_training_progress(history)
    
    return net, history


def main():
    folder = r"data\all"
    arrs, names = load_rle_files(folder, target_shape=(1000,1000), n_workers=20, device='cuda') 
    net, _ = train_patchnet(arrs, env_size=7, patch_size=3, epochs=5, batch_size=128)
    torch.save(net.state_dict(), r'data\network_final.pth')   