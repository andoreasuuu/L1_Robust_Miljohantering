"""
Lab 1 – Environment verification script
Run on any computer without modification
"""

import platform
import sys

import torch
import sklearn
import pandas
import jupyter_core


def detect_device():
    """Detect best available compute device"""
    if torch.cuda.is_available():
        device = "cuda"
        backend = "ROCm" if torch.version.hip else "CUDA"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        backend = "MPS"
    else:
        device = "cpu"
        backend = "CPU"

    return device, backend


def main():
    print("Lab 1: Environment check\n")

    print("System")
    print("Python:", sys.version.split()[0])
    print("Platform:", platform.platform())
    print()

    print("Package versions")
    print("torch:", torch.__version__)
    print("scikit-learn:", sklearn.__version__)
    print("pandas:", pandas.__version__)
    print("jupyter_core:", jupyter_core.__version__)
    print("torch CUDA build version:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())
    print()

    device, backend = detect_device()
    print("Accelerator status")
    print(f"Using device: {device} ({backend})")

    if device == "cuda":
        idx = torch.cuda.current_device()
        print("CUDA device index:", idx)
        print("CUDA device name:", torch.cuda.get_device_name(idx))
    print()

    print("Tensor computation test")
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device)
    y = torch.tensor([[5.0, 6.0], [7.0, 8.0]], device=device)

    z = x @ y
    expected = torch.tensor([[19.0, 22.0], [43.0, 50.0]], device=device)

    ok = torch.allclose(z, expected)
    print("Result correct:", ok)

    if not ok:
        raise SystemExit("Tensor computation failed!")

    print("\nAll checks passed ✅")


if __name__ == "__main__":
    main()
