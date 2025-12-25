import os
import numpy as np
import torch
import torch.nn as nn
import trimesh
from scipy.ndimage import gaussian_filter
from skimage import measure

# =========================================================
# MODEL DEFINITIONS (MUST MATCH TRAINING EXACTLY)
# =========================================================

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
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
        z_q = z_perm + (z_q - z_perm).detach()
        z_q = z_q.permute(0, 4, 1, 2, 3).contiguous()

        return z_q, unique_codes


class Encoder3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(1, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv3d(32, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv3d(64, 128, 3, 1, 1),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class Decoder3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose3d(128, 64, 3, 1, 1),
            nn.ReLU(),
            nn.ConvTranspose3d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose3d(32, 1, 4, 2, 1),
        )

    def forward(self, x):
        return self.net(x)


class VQAE3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder3D()
        self.vq = VectorQuantizer(2048, 128)
        self.decoder = Decoder3D()

    def forward(self, x):
        z = self.encoder(x)
        z_q, code_usage = self.vq(z)
        recon = self.decoder(z_q)
        return recon, code_usage


# =========================================================
# LOAD MODEL + DATA
# =========================================================

def main():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    MODEL_PATH = "./bitspace_book_vqae_epoch_80.pt"
    SDF_PATH = "./sdf_test_book/roadside_picnic_novel_glb (1).npy"

    model = VQAE3D().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # ================= ORIGINAL (GROUND TRUTH) =================
    # sdf_orig  -> original uncompressed SDF
    # mesh_orig -> ground-truth reference mesh

    sdf_orig = np.load(SDF_PATH).astype(np.float32)
    sdf_tensor = torch.from_numpy(sdf_orig)[None, None].to(DEVICE)

    # ================= RECONSTRUCTED (COMPRESSED) ==============
    # sdf_recon  -> decoded SDF from compressed latent
    # mesh_recon -> reconstructed mesh after compression

    sdf_in = torch.clamp(sdf_tensor, -0.1, 0.1)

    with torch.no_grad():
        recon, code_usage = model(sdf_in)

    sdf_recon = recon.cpu().numpy().squeeze()

    print(f"Codes used: {code_usage} / 2048")

    # =========================================================
    # METRICS (UNCHANGED)
    # =========================================================

    l1 = np.mean(np.abs(sdf_orig - sdf_recon))
    mse = np.mean((sdf_orig - sdf_recon) ** 2)

    surface_mask = np.abs(sdf_orig) < 0.02
    surface_l1 = np.mean(np.abs(sdf_orig[surface_mask] - sdf_recon[surface_mask]))

    print("\n=== Reconstruction Metrics ===")
    print(f"L1 Error        : {l1:.6f}")
    print(f"MSE             : {mse:.6f}")
    print(f"Surface L1      : {surface_l1:.6f}")

    # =========================================================
    # DEMO-ONLY VISUAL IMPROVEMENTS (RECONSTRUCTED ONLY)
    # =========================================================

    print("\nApplying demo-only SDF smoothing (RECONSTRUCTED)...")
    sdf_clean = gaussian_filter(sdf_recon, sigma=0.9)
    sdf_clean = np.clip(sdf_clean, -0.25, 0.25)

    # =========================================================
    # SURFACE EXTRACTION
    # =========================================================

    print("Extracting surface (RECONSTRUCTED)...")

    verts, faces, normals, _ = measure.marching_cubes(
        sdf_clean,
        level=0.0,
        spacing=(1 / 64, 1 / 64, 1 / 64)
    )

    mesh_recon = trimesh.Trimesh(
        verts,
        faces,
        vertex_normals=normals,
        process=False
    )

    # =========================================================
    # POST-PROCESSING (TRIMESH 4.x SAFE)
    # =========================================================

    print("Post-processing RECONSTRUCTED mesh...")

    mask = mesh_recon.nondegenerate_faces()
    mesh_recon.update_faces(mask)

    faces_sorted = np.sort(mesh_recon.faces, axis=1)
    _, unique_idx = np.unique(faces_sorted, axis=0, return_index=True)
    mesh_recon.update_faces(unique_idx)

    mesh_recon.remove_unreferenced_vertices()

    mesh_recon = trimesh.smoothing.filter_taubin(
        mesh_recon,
        lamb=0.5,
        nu=-0.53,
        iterations=3
    )

    mesh_recon.fix_normals()

    # =========================================================
    # ORIGINAL MESH CREATION (NO SMOOTHING)
    # =========================================================

    def sdf_to_mesh_raw(sdf):
        v, f, _, _ = measure.marching_cubes(
            sdf,
            level=0.0,
            spacing=(1 / 64, 1 / 64, 1 / 64)
        )
        return trimesh.Trimesh(v, f, process=False)

    mesh_orig = sdf_to_mesh_raw(sdf_orig)

    # =========================================================
    # EXPORT + LABELED SIDE-BY-SIDE VIEW
    # =========================================================
    mesh_orig.export("original_roadside_picnic_novel_demo.glb")
    mesh_recon.export("reconstructed_roadside_picnic_novel_demo.glb")
    print("\nSaved mesh → reconstructed_roadside_picnic_novel_demo.glb")

    # Color coding for clarity
    mesh_orig.visual.face_colors = [50, 200, 50, 255]          # GREEN = ORIGINAL
    mesh_recon.visual.face_colors = [200, 50, 50, 255]        # RED = RECONSTRUCTED

    bbox = mesh_orig.bounds
    width = bbox[1][0] - bbox[0][0]

    mesh_recon_shifted = mesh_recon.copy()
    mesh_recon_shifted.apply_translation([width * 1.6, 0, 0])

    scene = trimesh.Scene()
    scene.add_geometry(mesh_orig, node_name="ORIGINAL (Ground Truth)")
    scene.add_geometry(mesh_recon_shifted, node_name="RECONSTRUCTED (Compressed)")

    print("Opening side-by-side viewer...")
    scene.show()


if __name__ == "__main__":
    main()
