#!/usr/bin/env python3

import argparse
import os
import sys
import numpy as np
import torch
import trimesh
from scipy.ndimage import gaussian_filter
from skimage import measure

# ------------------------------
# Import BitSpace components
# ------------------------------
from model import VQAE3D
from to_sdf import glb_to_sdf

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "./bitspace_book_vqae_epoch_80.pt"

# ------------------------------
# Utilities
# ------------------------------

def load_model():
    model = VQAE3D().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model


def save_latent(latent, input_path):
    out = os.path.splitext(input_path)[0] + ".latent.npy"
    np.save(out, latent)
    print(f"[BitSpace] Saved latent → {out}")


def save_sdf(sdf, latent_path):
    out = latent_path.replace(".latent.npy", "_recon.npy")
    np.save(out, sdf)
    print(f"[BitSpace] Saved reconstructed SDF → {out}")


def save_glb(sdf, latent_path):
    out = latent_path.replace(".latent.npy", "_recon.glb")

    # Demo-safe smoothing
    sdf = gaussian_filter(sdf, sigma=0.9)
    sdf = np.clip(sdf, -0.25, 0.25)

    verts, faces, _, _ = measure.marching_cubes(
        sdf, level=0.0, spacing=(1 / 64, 1 / 64, 1 / 64)
    )

    mesh = trimesh.Trimesh(verts, faces, process=False)
    mesh.fix_normals()
    mesh.export(out)

    print(f"[BitSpace] Saved reconstructed mesh → {out}")

# ------------------------------
# COMMAND: compress
# ------------------------------

def compress_cmd(args):
    input_path = args.input

    if not os.path.exists(input_path):
        print(f"[BitSpace] File not found: {input_path}")
        sys.exit(1)

    ext = os.path.splitext(input_path)[1].lower()
    model = load_model()

    if ext == ".npy":
        sdf = np.load(input_path).astype(np.float32)

    elif ext == ".glb":
        print("[BitSpace] Extracting geometry from GLB...")
        sdf = glb_to_sdf(input_path)

    else:
        print("[BitSpace] Unsupported input format")
        sys.exit(1)

    sdf_tensor = torch.from_numpy(sdf)[None, None].to(DEVICE)
    sdf_tensor = torch.clamp(sdf_tensor, -0.1, 0.1)

    with torch.no_grad():
        latent = model.encoder(sdf_tensor)

    latent_np = latent.cpu().numpy()
    save_latent(latent_np, input_path)

    print("[BitSpace] Compression complete.")
    print(f"[BitSpace] Latent shape: {latent_np.shape}")

# ------------------------------
# COMMAND: decompress
# ------------------------------

def decompress_cmd(args):
    latent_path = args.latent

    if not os.path.exists(latent_path):
        print(f"[BitSpace] File not found: {latent_path}")
        sys.exit(1)

    model = load_model()

    latent = np.load(latent_path)
    latent_tensor = torch.from_numpy(latent).to(DEVICE)

    with torch.no_grad():
        sdf_recon = model.decoder(latent_tensor)

    sdf_recon = sdf_recon.cpu().numpy().squeeze()

    if args.to_glb:
        save_glb(sdf_recon, latent_path)
    else:
        save_sdf(sdf_recon, latent_path)

    print("[BitSpace] Decompression complete.")

# ------------------------------
# COMMAND: view
# ------------------------------

def view_cmd(args):
    path = args.input

    if not os.path.exists(path):
        print(f"[BitSpace] File not found: {path}")
        sys.exit(1)

    ext = os.path.splitext(path)[1].lower()
    print("[BitSpace] Viewer opened (verification only)")

    if ext == ".glb":
        mesh = trimesh.load(path, force="mesh")
        mesh.show()

    elif ext == ".npy":
        sdf = np.load(path).astype(np.float32)

        sdf = gaussian_filter(sdf, sigma=0.5)

        verts, faces, _, _ = measure.marching_cubes(
            sdf, level=0.0, spacing=(1 / 64, 1 / 64, 1 / 64)
        )

        mesh = trimesh.Trimesh(verts, faces, process=False)
        mesh.fix_normals()
        mesh.show()

    else:
        print("[BitSpace] Unsupported format for view")
        sys.exit(1)

# ------------------------------
# CLI Entry Point
# ------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="BitSpace CLI – Geometry Compression for AI Training"
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # compress
    compress_parser = subparsers.add_parser(
        "compress", help="Compress a 3D model into BitSpace latents"
    )
    compress_parser.add_argument("input", help=".glb mesh or .npy SDF")
    compress_parser.set_defaults(func=compress_cmd)

    # decompress
    decompress_parser = subparsers.add_parser(
        "decompress", help="Reconstruct geometry from BitSpace latents"
    )
    decompress_parser.add_argument("latent", help=".latent.npy file")
    decompress_parser.add_argument(
        "--to_sdf", action="store_true", help="Output reconstructed SDF (.npy)"
    )
    decompress_parser.add_argument(
        "--to_glb", action="store_true", help="Output reconstructed mesh (.glb)"
    )
    decompress_parser.set_defaults(func=decompress_cmd)

    # view
    view_parser = subparsers.add_parser(
        "view", help="View a mesh or SDF (verification only)"
    )
    view_parser.add_argument("input", help=".glb mesh or .npy SDF")
    view_parser.set_defaults(func=view_cmd)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
