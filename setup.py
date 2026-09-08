"""
SAMesh Mesh Segmentation -- Modly extension setup script.

Called by Modly at install time:
    python setup.py <json_args>

json_args keys:
    python_exe  - path to Modly's embedded Python
    ext_dir     - absolute path to this extension directory
    gpu_sm      - GPU compute capability as integer (e.g. 89 for RTX 4090)
"""
import json
import os
import platform
import subprocess
import sys
from pathlib import Path


IS_WIN = platform.system() == "Windows"


def pip(venv, *args):
    pip_exe = venv / ("Scripts/pip.exe" if IS_WIN else "bin/pip")
    subprocess.run([str(pip_exe)] + list(args), check=True)


def python_exe_in_venv(venv):
    return venv / ("Scripts/python.exe" if IS_WIN else "bin/python")


def uninstall(ext_dir):
    import shutil
    import stat

    def _rm_readonly(func, path, exc_info):
        try:
            os.chmod(path, stat.S_IWRITE)
            func(path)
        except Exception:
            pass

    for target_name in ("venv", "samesh"):
        target = ext_dir / target_name
        if target.exists():
            print("[uninstall] Removing %s ..." % target)
            shutil.rmtree(str(target), onerror=_rm_readonly)
            print("[uninstall] Removed %s." % target_name)
        else:
            print("[uninstall] %s not found, skipping." % target_name)

    print("[uninstall] Done.")


def setup(python_exe, ext_dir, gpu_sm):
    venv = ext_dir / "venv"

    if not venv.exists():
        print("[setup] Creating venv at %s ..." % venv)
        subprocess.run([str(python_exe), "-m", "venv", str(venv)], check=True)
    else:
        print("[setup] Venv exists, skipping creation.")

    venv_python = python_exe_in_venv(venv)

    # ------------------------------------------------------------------ #
    # Build prerequisites
    # ------------------------------------------------------------------ #
    print("[setup] Installing build prerequisites...")
    pip(venv, "install", "ninja", "setuptools", "wheel")

    # ------------------------------------------------------------------ #
    # PyTorch -- same arch tiers as the hunyuan3d-part extension
    # ------------------------------------------------------------------ #
    if gpu_sm >= 100:
        torch_index = "https://download.pytorch.org/whl/cu128"
        torch_pkgs = ["torch>=2.7.0", "torchvision>=0.22.0", "torchaudio>=2.7.0"]
        print("[setup] SM %d (Blackwell) -> PyTorch 2.7 + CUDA 12.8" % gpu_sm)
    elif gpu_sm >= 70:
        torch_index = "https://download.pytorch.org/whl/cu124"
        torch_pkgs = ["torch==2.6.0", "torchvision==0.21.0", "torchaudio==2.6.0"]
        print("[setup] SM %d -> PyTorch 2.6.0 + CUDA 12.4" % gpu_sm)
    else:
        torch_index = "https://download.pytorch.org/whl/cu118"
        torch_pkgs = ["torch==2.5.1", "torchvision==0.20.1", "torchaudio==2.5.1"]
        print("[setup] SM %d (legacy) -> PyTorch 2.5.1 + CUDA 11.8" % gpu_sm)

    print("[setup] Installing PyTorch...")
    pip(venv, "install", *torch_pkgs, "--index-url", torch_index)

    # ------------------------------------------------------------------ #
    # Core dependencies (SAMesh + SAM2)
    # ------------------------------------------------------------------ #
    print("[setup] Installing core dependencies...")
    pip(venv, "install",
        "numpy==1.26.4",
        "trimesh",
        "pymeshlab",
        "omegaconf",
        "igraph",
        "networkx",
        "pyrender",
        "PyOpenGL==3.1.7",
        "scikit-learn",
        "pandas",
        "natsort",
        "matplotlib",
        "opencv-python",
        "transformers",
        "lightning",
        "torchtyping",
        "tqdm",
        "einops",
        "huggingface_hub",
        "Pillow",
    )

    # ------------------------------------------------------------------ #
    # Clone SAMesh repo (with submodules for SAM2)
    # ------------------------------------------------------------------ #
    samesh_dir = ext_dir / "samesh"
    if not samesh_dir.exists():
        print("[setup] Cloning SAMesh repo...")
        cloned = False
        for url in [
            "https://github.com/gtangg12/samesh.git",
        ]:
            try:
                subprocess.run(
                    ["git", "clone", "--depth=1", "--recursive", url, str(samesh_dir)],
                    check=True,
                    timeout=300,
                )
                cloned = True
                print("[setup] Cloned from %s" % url)
                break
            except subprocess.CalledProcessError as e:
                print("[setup] Clone from %s failed: %s" % (url, e))
            except subprocess.TimeoutExpired:
                print("[setup] Clone from %s timed out." % url)
        if not cloned:
            print(
                "[setup] WARNING: Could not clone SAMesh repo.\n"
                "[setup]   Download manually from https://github.com/gtangg12/samesh\n"
                "[setup]   and place it at: %s" % samesh_dir
            )
    else:
        print("[setup] SAMesh repo already exists, skipping clone.")

    # ------------------------------------------------------------------ #
    # Patch pyproject.toml to allow Python 3.11
    # ------------------------------------------------------------------ #
    pyproject = samesh_dir / "pyproject.toml"
    if pyproject.exists():
        try:
            content = pyproject.read_text(encoding="utf-8")
            patched = content.replace(
                'requires-python = ">=3.12.0"',
                'requires-python = ">=3.11.0"',
            )
            if patched != content:
                pyproject.write_text(patched, encoding="utf-8")
                print("[setup] Patched pyproject.toml for Python 3.11+.")
            else:
                print("[setup] pyproject.toml already allows 3.11+ or format changed.")
        except Exception as e:
            print("[setup] WARNING: Could not patch pyproject.toml: %s" % e)

    # ------------------------------------------------------------------ #
    # Install SAMesh from source (editable, skip deps since we installed them)
    # ------------------------------------------------------------------ #
    if samesh_dir.exists():
        print("[setup] Installing SAMesh package...")
        subprocess.run(
            [str(venv_python), "-m", "pip", "install", "-e", str(samesh_dir),
             "--no-deps", "--no-build-isolation"],
            check=False,
        )

    # ------------------------------------------------------------------ #
    # Install SAM2 (segment-anything-2) if submodule exists
    # ------------------------------------------------------------------ #
    sam2_dir = samesh_dir / "third_party" / "segment-anything-2"
    if sam2_dir.exists():
        print("[setup] Installing SAM2 from submodule...")
        subprocess.run(
            [str(venv_python), "-m", "pip", "install", "-e", str(sam2_dir),
             "--no-build-isolation"],
            check=False,
        )
    else:
        print("[setup] SAM2 submodule not found, installing from GitHub...")
        try:
            subprocess.run(
                [str(venv_python), "-m", "pip", "install",
                 "git+https://github.com/facebookresearch/sam2.git"],
                check=True,
                timeout=180,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            print("[setup] WARNING: SAM2 install failed: %s" % e)
            print("[setup]   You may need to install SAM2 manually.")

    # ------------------------------------------------------------------ #
    # Download SAM2 checkpoints
    # ------------------------------------------------------------------ #
    # Try MODELS_DIR env var first (Modly sets this for generator processes),
    # then fall back to the standard modlydata/models/<extension-id> layout.
    _models_base = os.environ.get("MODELS_DIR") or str(ext_dir.parent.parent / "models")
    models_dir = Path(_models_base) / "samesh-mesh-seg"
    models_dir.mkdir(parents=True, exist_ok=True)
    print("[setup] Model weights dir: %s" % models_dir)
    print("[setup] MODELS_DIR env: %s" % os.environ.get("MODELS_DIR", "(not set)"))

    for size, repo_id, filename in [
        ("small", "facebook/sam2-hiera-small", "sam2_hiera_small.pt"),
        ("large", "facebook/sam2-hiera-large", "sam2_hiera_large.pt"),
    ]:
        target = models_dir / filename
        if not target.exists():
            print("[setup] Downloading SAM2 %s checkpoint..." % size)
            try:
                subprocess.run(
                    [str(venv_python), "-c",
                     "from huggingface_hub import hf_hub_download; "
                     "hf_hub_download(repo_id='%s', filename='%s', local_dir='%s')"
                     % (repo_id, filename, models_dir)],
                    check=True,
                    timeout=300,
                )
                print("[setup] SAM2 %s downloaded." % size)
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                print("[setup] WARNING: SAM2 %s download failed: %s" % (size, e))
        else:
            print("[setup] SAM2 %s already exists, skipping." % size)

    # ------------------------------------------------------------------ #
    # Verify imports
    # ------------------------------------------------------------------ #
    print("[setup] Verifying imports...")
    check = subprocess.run(
        [str(venv_python), "-c",
         "import torch; print('torch:', torch.__version__, '| CUDA:', torch.cuda.is_available()); "
         "import trimesh; print('trimesh:', trimesh.__version__); "
         "import pyrender; print('pyrender: ok'); "
         "import omegaconf; print('omegaconf:', omegaconf.__version__)"],
        capture_output=True, text=True,
    )
    if check.returncode == 0:
        for line in check.stdout.strip().splitlines():
            print("[setup] %s" % line)
    else:
        print("[setup] WARNING: import check failed:\n%s" % check.stderr.strip())

    print("[setup] Done. Venv ready at: %s" % venv)


if __name__ == "__main__":
    if len(sys.argv) >= 4:
        setup(
            python_exe=Path(sys.argv[1]),
            ext_dir=Path(sys.argv[2]),
            gpu_sm=int(sys.argv[3]),
        )
    elif len(sys.argv) == 2:
        args = json.loads(sys.argv[1])
        action = args.get("action", "install")
        if action == "uninstall":
            uninstall(Path(args["ext_dir"]))
        else:
            setup(
                python_exe=Path(args["python_exe"]),
                ext_dir=Path(args["ext_dir"]),
                gpu_sm=int(args["gpu_sm"]),
            )
    else:
        print("Usage: python setup.py <python_exe> <ext_dir> <gpu_sm>")
        print('   or: python setup.py \'{"python_exe":"...","ext_dir":"...","gpu_sm":89}\'')
        sys.exit(1)
