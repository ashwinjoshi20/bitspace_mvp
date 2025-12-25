import os
import gc
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# =========================================================
# DATASET
# =========================================================

class SDFDataset(Dataset):
    def __init__(self, sdf_dir):
        self.files = [
            os.path.join(sdf_dir, f)
            for f in os.listdir(sdf_dir)
            if f.endswith(".npy")
        ]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        sdf = np.load(self.files[idx]).astype(np.float32)
        sdf = torch.from_numpy(sdf)
        return sdf.unsqueeze(0)  # (1, 64, 64, 64)


# =========================================================
# VECTOR QUANTIZER
# =========================================================

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(
            -1 / num_embeddings, 1 / num_embeddings
        )

    def forward(self, z):
        z_perm = z.permute(0, 2, 3, 4, 1).contiguous()
        flat_z = z_perm.view(-1, self.embedding_dim)

        distances = (
            flat_z.pow(2).sum(1, keepdim=True)
            - 2 * flat_z @ self.embedding.weight.t()
            + self.embedding.weight.pow(2).sum(1)
        )

        encoding_indices = torch.argmin(distances, dim=1)
        unique_codes = torch.unique(encoding_indices).numel()

        z_q = self.embedding(encoding_indices).view(z_perm.shape)

        commitment_loss = F.mse_loss(z_q.detach(), z_perm)
        codebook_loss = F.mse_loss(z_q, z_perm.detach())
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss

        # straight-through estimator
        z_q = z_perm + (z_q - z_perm).detach()
        z_q = z_q.permute(0, 4, 1, 2, 3).contiguous()

        return z_q, vq_loss, unique_codes


# =========================================================
# ENCODER (64³ → 16³)
# =========================================================

class Encoder3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(1, 32, 4, 2, 1),    # 32³
            nn.ReLU(),
            nn.Conv3d(32, 64, 4, 2, 1),   # 16³
            nn.ReLU(),
            nn.Conv3d(64, 128, 3, 1, 1),  # ✅ FIXED (stay 16³)
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


# =========================================================
# DECODER (16³ → 64³)
# =========================================================

class Decoder3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose3d(128, 64, 3, 1, 1),  # ✅ FIXED (stay 16³)
            nn.ReLU(),
            nn.ConvTranspose3d(64, 32, 4, 2, 1),   # 32³
            nn.ReLU(),
            nn.ConvTranspose3d(32, 1, 4, 2, 1),    # 64³
        )

    def forward(self, x):
        return self.net(x)


# =========================================================
# VQ-AE MODEL
# =========================================================

class VQAE3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder3D()
        self.vq = VectorQuantizer(
            num_embeddings=2048,   # ✅ FIXED
            embedding_dim=128
        )
        self.decoder = Decoder3D()

    def forward(self, x):
        z = self.encoder(x)
        z_q, vq_loss, code_usage = self.vq(z)
        recon = self.decoder(z_q)
        return recon, vq_loss, code_usage


# =========================================================
# TRAINING SETUP
# =========================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_DIR = "./sdf_book"   # change if needed
EPOCHS = 80
dx = 1.0 / 64

dataset = SDFDataset(DATA_DIR)
loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=True,
    num_workers=0   # ✅ Windows-safe
)

model = VQAE3D().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)


# =========================================================
# TRAINING LOOP
# =========================================================

for epoch in range(EPOCHS):
    total_loss = 0.0
    total_codes = 0

    for sdf in tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):

        sdf_gt = sdf.to(DEVICE)
        sdf_in = torch.clamp(sdf_gt, -0.1, 0.1)

        recon, vq_loss, code_usage = model(sdf_in)

        # surface-weighted L1
        weight = torch.exp(-torch.abs(sdf_gt) / 0.02)
        recon_loss = (weight * torch.abs(recon - sdf_gt)).mean()

        # Eikonal loss
        grads = torch.gradient(
            recon,
            spacing=(dx, dx, dx),
            dim=(2, 3, 4)
        )
        grad_norm = torch.sqrt(sum(g**2 for g in grads) + 1e-8)
        eikonal_loss = ((grad_norm - 1) ** 2 * weight).mean()

        loss = recon_loss + vq_loss + 0.1 * eikonal_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_codes += code_usage

        del sdf_gt, sdf_in, recon
        gc.collect()

    print(
        f"Epoch {epoch+1} | "
        f"Loss: {total_loss / len(loader):.4f} | "
        f"Codes/sample: {total_codes / len(loader):.1f}"
    )

    if (epoch + 1) % 10 == 0:
        torch.save(
            model.state_dict(),
            f"bitspace_book_vqae_epoch_{epoch+1}.pt"
        )
