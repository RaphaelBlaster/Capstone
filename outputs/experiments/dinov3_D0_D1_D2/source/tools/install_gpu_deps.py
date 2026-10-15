"""Preserve the host CUDA torch build and select its matching torchao release."""
from pathlib import Path
import argparse
import subprocess
import sys

import torch
from packaging.version import Version

version = Version(torch.__version__.split("+")[0])
# Official compatibility table checked 2026-10-01: github.com/pytorch/ao/issues/2919.
# Float8Linear uses the Python API + native torch scaled_mm, not torchao's CUTLASS extensions.
if version >= Version("2.11"):
    ao_version = "0.17.0"
elif version >= Version("2.10"):
    ao_version = "0.16.0"
elif version >= Version("2.9.1"):
    ao_version = "0.15.0"
elif version >= Version("2.9"):
    ao_version = "0.14.1"
elif version >= Version("2.8"):
    ao_version = "0.13.0"
elif version >= Version("2.7.1"):
    ao_version = "0.12.0"
else:
    raise SystemExit("GPU runner requires torch >= 2.7.1. Select a recent CUDA GPU image first.")

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--wheels", type=Path, help="Offline mounted wheel directory; no network requests permitted.")
args = parser.parse_args()
print(f"Keeping torch {torch.__version__}; installing torchao {ao_version}", flush=True)
if args.wheels:
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps",
                    "--find-links", str(args.wheels), "timm==1.0.30", f"torchao=={ao_version}",
                    "huggingface-hub==0.36.2", "safetensors", "tqdm", "packaging"], check=True)
else:
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(root/"requirements.txt"),
                    f"torchao=={ao_version}"], check=True)
