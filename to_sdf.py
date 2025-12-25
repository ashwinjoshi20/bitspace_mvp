import os
import gc
import numpy as np
import trimesh
from mesh_to_sdf import mesh_to_voxels
from tqdm import tqdm

# INPUT: augmented GLBs
INPUT_DIR = "./test"
OUTPUT_DIR = "./sdf_test_book"

os.makedirs(OUTPUT_DIR, exist_ok=True)

SDF_RESOLUTION = 64

def normalize_mesh(mesh):
    mesh = mesh.copy()
    mesh.vertices -= mesh.bounding_box.centroid
    scale = np.max(mesh.bounding_box.extents)
    mesh.vertices /= scale
    return mesh

def load_mesh_from_glb(glb_path):
    scene = trimesh.load(glb_path, force="scene")
    if not scene.geometry:
        raise ValueError("No geometry found in GLB")
    mesh = trimesh.util.concatenate(tuple(scene.geometry.values()))
    del scene
    gc.collect()
    return mesh

def glb_to_sdf(glb_path):
    mesh = load_mesh_from_glb(glb_path)
    mesh = normalize_mesh(mesh)

    # mesh_to_voxels returns a 3D numpy array of signed distances
    sdf_grid = mesh_to_voxels(mesh, voxel_resolution=SDF_RESOLUTION).astype(np.float32)

    del mesh
    gc.collect()
    return sdf_grid

def main():
    glb_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(".glb")]
    print(f"[BitSpace] Found {len(glb_files)} GLB files")

    for idx, glb_file in enumerate(glb_files, start=1):
        glb_path = os.path.join(INPUT_DIR, glb_file)
        out_path = os.path.join(OUTPUT_DIR, glb_file.replace(".glb", ".npy"))

        try:
            print(f"[{idx}/{len(glb_files)}] Converting {glb_file}")
            sdf = glb_to_sdf(glb_path)
            np.save(out_path, sdf)
            print(f"    ✓ Saved {out_path}")
            del sdf
        except Exception as e:
            print(f"    ✗ Failed: {e}")

        gc.collect()

    print("[BitSpace] SDF conversion complete.")


if __name__ == "__main__":
    main()