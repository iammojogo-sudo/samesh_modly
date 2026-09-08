# SAMesh Mesh Segmentation

Zero-shot 3D mesh part segmentation using [SAMesh](https://github.com/gtangg12/samesh) + [SAM2](https://github.com/facebookresearch/sam2). Splits a mesh into semantically meaningful parts without any training.

## What it does

Takes a single GLB mesh and segments it into separate part objects. For example, a chair mesh gets split into seat, back, legs, armrests — automatically, with no labels or training data required.

## How it works

1. Renders the mesh from multiple angles in different modalities (normals, shape diameter function, matte)
2. Runs SAM2 on each rendered view to generate 2D masks
3. Lifts the 2D masks into a consistent 3D face-level segmentation
4. Smooths boundaries and removes artifacts
5. Exports each segment as a separate mesh object in a GLB scene

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| SAM2 Model | Small (~40 MB) | Small is fast, Large (~2.5 GB) is more precise |
| Render Modalities | SDF + Normals | Which features to render. Add Matte for low-poly meshes |
| Render Resolution | 1024 | Higher = better but slower |
| SAM Mask Density | 32 | Points per side for SAM2 mask generation |
| Smoothing | 64 | Boundary smoothing iterations |

## Tips

- **Simple shapes** (chairs, tables, mugs): SDF + Normals at 1024 works great
- **Low-poly / stylized meshes**: Try SDF + Normals + Matte
- **Very detailed meshes**: Increase mask density to 64
- **Fast preview**: Use Small SAM2 model + 512 resolution

## Dependencies

- PyTorch (CUDA)
- SAM2 (Facebook)
- SAMesh (gtangg12)
- pyrender + PyOpenGL
- trimesh, pymeshlab, omegaconf

## License

MIT — see LICENSE file. SAMesh and SAM2 have their own licenses.
