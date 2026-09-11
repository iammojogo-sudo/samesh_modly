import hashlib
import io
import json
import os
import sys
import tempfile
import time
import uuid
import gc
import zipfile
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import torch
import trimesh

from services.generators.base import BaseGenerator, smooth_progress

_print = print
def print(*args, **kwargs):
    kwargs.setdefault("file", sys.stderr)
    _print(*args, **kwargs)

try:
    sys.stdout.reconfigure(errors="replace")
    sys.stderr.reconfigure(errors="replace")
except (AttributeError, io.UnsupportedOperation):
    pass

_LOG = "[SAMeshGenerator]"

_MODALITY_MAP = {
    "sdf_norms": ["sdf", "norms"],
    "sdf_norms_matte": ["sdf", "norms", "matte"],
    "norms_matte": ["norms", "matte"],
    "matte": ["matte"],
}

_SAM2_CONFIG_MAP = {
    "small": ("sam2_hiera_s.yaml", "sam2_hiera_small.pt"),
    "large": ("sam2_hiera_l.yaml", "sam2_hiera_large.pt"),
}

_SAM2_HF_MAP = {
    "small": "facebook/sam2-hiera-small",
    "large": "facebook/sam2-hiera-large",
}


def _safe_int(val, default):
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


def _cuda_total_gb():
    try:
        if torch.cuda.is_available():
            return float(torch.cuda.get_device_properties(0).total_mem) / (1024 ** 3)
    except Exception:
        pass
    return 0.0


_ES_CONTINUOUS = 0x80000000
_ES_SYSTEM_REQUIRED = 0x00000001


def _keep_system_awake(enable):
    try:
        import ctypes
        if os.name == "nt":
            state = _ES_CONTINUOUS | (_ES_SYSTEM_REQUIRED if enable else 0)
            ctypes.windll.kernel32.SetThreadExecutionState(state)
    except Exception:
        pass


def _setup_opengl_platform():
    """Try to set up OpenGL platform for pyrender on Windows."""
    if "PYOPENGL_PLATFORM" in os.environ:
        return
    # On Windows, don't set PYOPENGL_PLATFORM — let PyOpenGL use
    # the native opengl32.dll driver (WGL). EGL/OSMesa are Linux-only.
    if os.name == "nt":
        print("%s Using default Windows OpenGL backend." % _LOG)


class SAMeshGenerator(BaseGenerator):
    MODEL_ID = "samesh-mesh-seg"
    DISPLAY_NAME = "SAMesh Mesh Segmentation"
    VRAM_GB = 6

    def _find_checkpoint(self, size):
        """Find a SAM2 checkpoint file, searching multiple locations."""
        _, filename = _SAM2_CONFIG_MAP[size]

        # 1. Check model_dir directly
        p = self.model_dir / filename
        if p.exists():
            return p

        # 2. Check subdirectories of model_dir (Modly may nest downloads)
        for child in self.model_dir.iterdir():
            if child.is_dir():
                p = child / filename
                if p.exists():
                    return p

        # 3. Check models/ relative to extension dir (modlydata layout)
        ext_models = Path(__file__).parent.parent.parent / "models" / self.MODEL_ID
        if ext_models.exists():
            p = ext_models / filename
            if p.exists():
                return p
            for child in ext_models.iterdir():
                if child.is_dir():
                    p = child / filename
                    if p.exists():
                        return p

        # 4. Check MODELS_DIR env var
        models_dir_env = os.environ.get("MODELS_DIR")
        if models_dir_env:
            p = Path(models_dir_env) / self.MODEL_ID / filename
            if p.exists():
                return p
            base = Path(models_dir_env) / self.MODEL_ID
            if base.exists():
                for child in base.iterdir():
                    if child.is_dir():
                        p = child / filename
                        if p.exists():
                            return p

        return None

    def is_downloaded(self) -> bool:
        small = self._find_checkpoint("small")
        large = self._find_checkpoint("large")
        return small is not None or large is not None

    def _ensure_samesh_on_path(self):
        samesh_dir = Path(__file__).parent / "samesh"
        src_dir = samesh_dir / "src"
        if not samesh_dir.exists():
            raise RuntimeError(
                "%s SAMesh repo not found at %s.\n"
                "Please reinstall or repair the extension." % (_LOG, samesh_dir)
            )
        if src_dir.exists() and str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))
        elif str(samesh_dir) not in sys.path:
            sys.path.insert(0, str(samesh_dir))

        sam2_dir = samesh_dir / "third_party" / "segment-anything-2"
        if sam2_dir.exists() and str(sam2_dir) not in sys.path:
            sys.path.insert(0, str(sam2_dir))

    def load(self):
        if self._model is not None:
            return
        if not self.is_downloaded():
            self._download_checkpoints()
        self._ensure_samesh_on_path()
        _setup_opengl_platform()
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = True
        print("%s Ready on %s." % (_LOG, self._device))

    def unload(self):
        self._model = None
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def generate(self, image_bytes, params, progress_cb=None, cancel_event=None):
        params = params or {}
        tmp_path = None
        try:
            mesh_path = None
            if isinstance(image_bytes, (bytes, bytearray)):
                data = image_bytes
                if data[:4] == b"glTF":
                    ext = ".glb"
                else:
                    ext = None
                    try:
                        head = data[:512].decode("utf-8", errors="ignore")
                        for line in head.splitlines()[:15]:
                            for tok in ("# ", "v ", "vt ", "vn ", "f ", "g ", "o ", "mtllib"):
                                if line.strip().startswith(tok):
                                    ext = ".obj"
                                    break
                            if ext:
                                break
                    except Exception:
                        pass
                if ext:
                    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
                        f.write(data)
                        tmp_path = f.name
                    mesh_path = tmp_path
            if mesh_path is None:
                for key in ("mesh_path", "input_mesh_path", "primary_mesh_path", "file_path"):
                    v = params.get(key)
                    if v and os.path.isfile(str(v)):
                        mesh_path = str(v)
                        break
            if mesh_path is None:
                raise RuntimeError(
                    "No mesh input found. Connect a GLB or OBJ mesh as input."
                )
            return self._segment(mesh_path, params, progress_cb, cancel_event)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

    def _segment(self, mesh_path, params, progress_cb=None, cancel_event=None):
        sam2_size = params.get("sam2_size", "small")
        if sam2_size not in _SAM2_CONFIG_MAP:
            sam2_size = "small"
        modality_key = params.get("render_modalities", "sdf_norms")
        use_modes = _MODALITY_MAP.get(modality_key, ["sdf", "norms"])
        render_res = _safe_int(params.get("render_resolution", 1024), 1024)
        points_per_side = _safe_int(params.get("points_per_side", 32), 32)
        smoothing_iters = _safe_int(params.get("smoothing_iterations", 64), 64)
        camera_method = params.get("camera_method", "icosahedron")
        use_cache = params.get("cache", False)

        model_config_name, checkpoint_name = _SAM2_CONFIG_MAP[sam2_size]
        checkpoint_path = self._find_checkpoint(sam2_size)

        if checkpoint_path is None:
            print("%s Checkpoint not found, downloading..." % _LOG)
            self._download_checkpoint(sam2_size)
            checkpoint_path = self._find_checkpoint(sam2_size)
            if checkpoint_path is None:
                raise RuntimeError(
                    "%s SAM2 %s checkpoint not found after download.\n"
                    "Check your internet connection and try again." % (_LOG, sam2_size)
                )

        print("%s params: sam2=%s modes=%s res=%d pps=%d smooth=%d cam=%s cache=%s"
              % (_LOG, sam2_size, use_modes, render_res, points_per_side,
                 smoothing_iters, camera_method, use_cache))

        _keep_system_awake(True)
        self._report(progress_cb, 5, "Loading SAMesh...")

        try:
            self.load()
            self._check_cancelled(cancel_event)

            from omegaconf import OmegaConf
            from samesh.models.sam_mesh import segment_mesh

            cache_dir = None
            if use_cache:
                cache_key = "%s_%d_%d_%s" % (
                    sam2_size, render_res, points_per_side,
                    "_".join(sorted(use_modes)),
                )
                cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:12]
                cache_dir = self.outputs_dir / "cache" / cache_hash
                cache_dir.mkdir(parents=True, exist_ok=True)
                print("%s Cache dir: %s" % (_LOG, cache_dir))

            config = OmegaConf.create({
                "cache": str(cache_dir) if cache_dir else None,
                "cache_overwrite": False,
                "output": str(self.outputs_dir),
                "sam": {
                    "sam": {
                        "checkpoint": str(checkpoint_path),
                        "model_config": model_config_name,
                        "auto": True,
                        "ground": False,
                        "engine_config": {
                            "points_per_side": points_per_side,
                            "crop_n_layers": 0,
                            "pred_iou_thresh": 0.5,
                            "stability_score_thresh": 0.7,
                            "stability_score_offset": 1.0,
                        },
                    },
                },
                "sam_mesh": {
                    "use_modes": use_modes,
                    "min_area": 1024,
                    "connections_bin_resolution": 100,
                    "connections_bin_threshold_percentage": 0.125,
                    "smoothing_threshold_percentage_size": 0.025,
                    "smoothing_threshold_percentage_area": 0.025,
                    "smoothing_iterations": smoothing_iters,
                    "repartition_cost": 1,
                    "repartition_lambda": 6,
                    "repartition_iterations": 1,
                },
                "renderer": {
                    "target_dim": [render_res, render_res],
                    "camera_generation_method": camera_method,
                    "renderer_args": {"interpolate_norms": True},
                    "sampling_args": {"radius": 2},
                    "lighting_args": {},
                },
            })

            self._report(progress_cb, 15, "Running SAMesh segmentation...")

            with torch.inference_mode():
                torch.cuda.empty_cache()
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    segmented_mesh = segment_mesh(
                        filename=mesh_path,
                        config=config,
                        visualize=False,
                        extension="glb",
                        texture=False,
                    )

            self._check_cancelled(cancel_event)
            self._report(progress_cb, 85, "Splitting into parts...")

            parts = self._split_by_labels(mesh_path, segmented_mesh)

            self._report(progress_cb, 95, "Exporting...")
            self.outputs_dir.mkdir(parents=True, exist_ok=True)
            timestamp = "%d_%s" % (int(time.time()), uuid.uuid4().hex[:8])
            scene_path = self.outputs_dir / ("%s_segmented.glb" % timestamp)
            zip_path = self.outputs_dir / ("%s_parts.zip" % timestamp)

            scene = trimesh.Scene()
            for i, part_mesh in enumerate(parts):
                scene.add_geometry(
                    part_mesh,
                    node_name="part_%d" % i,
                    geom_name="part_%d" % i,
                )
            scene.export(str(scene_path))
            print("%s Scene GLB: %s (%d parts)" % (_LOG, scene_path, len(parts)))

            with zipfile.ZipFile(str(zip_path), "w", zipfile.ZIP_DEFLATED) as zf:
                for i, part_mesh in enumerate(parts):
                    buf = io.BytesIO()
                    part_mesh.export(buf, file_type="glb")
                    zf.writestr("part_%d.glb" % i, buf.getvalue())
            print("%s Parts ZIP: %s" % (_LOG, zip_path))

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self._report(progress_cb, 100, "Done")
            return str(scene_path)

        finally:
            _keep_system_awake(False)

    def _split_by_labels(self, original_path, colored_mesh):
        """Split a colored trimesh into separate meshes per face label."""
        try:
            stem = Path(original_path).stem
            label_file = None
            for candidate in [
                self.outputs_dir / stem / "%s_face2label.json" % stem,
                self.outputs_dir / "%s_face2label.json" % stem,
            ]:
                if candidate.exists():
                    label_file = candidate
                    break

            if label_file is not None:
                with open(label_file, "r") as f:
                    raw_labels = json.load(f)
                face_labels = {int(k): int(v) for k, v in raw_labels.items()}
            else:
                face_colors = colored_mesh.visual.face_colors
                if face_colors is not None and len(face_colors) > 0:
                    unique_colors = np.unique(
                        face_colors[:, :3].reshape(-1, 3), axis=0
                    )
                    face_labels = {}
                    for fi in range(len(colored_mesh.faces)):
                        rgb = tuple(face_colors[fi, :3])
                        label = np.where(
                            np.all(unique_colors == rgb, axis=1)
                        )[0]
                        face_labels[fi] = int(label[0]) if len(label) > 0 else 0
                else:
                    face_labels = {i: 0 for i in range(len(colored_mesh.faces))}

        except Exception as e:
            print("%s Label split fallback: %s" % (_LOG, e))
            face_labels = {i: 0 for i in range(len(colored_mesh.faces))}

        label_to_faces = {}
        for fi, label in face_labels.items():
            label_to_faces.setdefault(label, []).append(fi)

        if not label_to_faces:
            return [colored_mesh]

        verts = colored_mesh.vertices
        faces = colored_mesh.faces
        parts = []

        for label in sorted(label_to_faces.keys()):
            face_indices = np.array(label_to_faces[label])
            sub_faces = faces[face_indices]

            used_verts = np.unique(sub_faces.ravel())
            vert_map = np.zeros(len(verts), dtype=np.int64)
            vert_map[used_verts] = np.arange(len(used_verts))

            new_faces = vert_map[sub_faces]
            new_verts = verts[used_verts]

            part_mesh = trimesh.Trimesh(vertices=new_verts, faces=new_faces, process=False)

            if colored_mesh.visual is not None and colored_mesh.visual.kind == "face":
                face_colors = colored_mesh.visual.face_colors
                if face_colors is not None and len(face_colors) > 0:
                    part_colors = face_colors[face_indices]
                    part_mesh.visual = trimesh.visual.ColorVisuals(
                        mesh=part_mesh,
                        face_colors=part_colors,
                    )

            parts.append(part_mesh)

        print("%s Split into %d parts." % (_LOG, len(parts)))
        return parts

    def _download_checkpoint(self, size="small"):
        """Download SAM2 checkpoint from HuggingFace."""
        _, checkpoint_name = _SAM2_CONFIG_MAP[size]
        hf_repo = _SAM2_HF_MAP[size]

        # Try model_dir first, then fallback locations
        target_dirs = [self.model_dir]
        ext_models = Path(__file__).parent.parent.parent / "models" / self.MODEL_ID
        if ext_models != self.model_dir:
            target_dirs.append(ext_models)

        for target_dir in target_dirs:
            target = target_dir / checkpoint_name
            if target.exists():
                print("%s SAM2 %s already at %s" % (_LOG, size, target))
                return target

        # Download to model_dir
        self.model_dir.mkdir(parents=True, exist_ok=True)
        print("%s Downloading SAM2 %s from %s..." % (_LOG, size, hf_repo))

        try:
            from huggingface_hub import hf_hub_download
            hf_hub_download(
                repo_id=hf_repo,
                filename=checkpoint_name,
                local_dir=str(self.model_dir),
            )
            print("%s SAM2 %s downloaded to %s" % (_LOG, size, self.model_dir))
            return self.model_dir / checkpoint_name
        except Exception as e:
            print("%s Download failed: %s" % (_LOG, e))
            raise RuntimeError(
                "%s Failed to download SAM2 %s checkpoint.\n"
                "Check your internet connection.\n"
                "Error: %s" % (_LOG, size, e)
            )

    def _download_checkpoints(self):
        """Download both small and large SAM2 checkpoints."""
        self.model_dir.mkdir(parents=True, exist_ok=True)
        for size in ("small", "large"):
            try:
                self._download_checkpoint(size)
            except Exception as e:
                print("%s Warning: Could not download SAM2 %s: %s" % (_LOG, size, e))
