"""Environment check for DC-CellNet. Pure Python; works on Windows/Linux/macOS.

Prints torch version, CUDA availability and GPU name. Exit code 1 if CUDA is unavailable.
"""
import sys


def main() -> int:
    try:
        import torch
    except ImportError as exc:
        print(f"torch import failed: {exc}")
        return 1
    print(f"torch {torch.__version__}")
    available = torch.cuda.is_available()
    print(available)
    if available:
        print(torch.cuda.get_device_name(0))
        return 0
    print("CUDA not available")
    return 1


if __name__ == "__main__":
    sys.exit(main())
