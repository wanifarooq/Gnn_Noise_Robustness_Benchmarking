"""Reproducibility helpers — seed all RNG sources for deterministic runs."""

import os
import random
import numpy as np
import torch


def setup_seed_device(seed: int):
    """Seed every RNG source used by the pipeline and enable deterministic CUDA ops.

    Sources seeded:
        - Python hash seed (PYTHONHASHSEED)
        - Python stdlib ``random``
        - NumPy global RNG
        - PyTorch CPU and all CUDA devices
        - cuDNN (deterministic mode, benchmark disabled)
        - PyTorch deterministic algorithms flag
        - cuBLAS workspace config (required by deterministic algorithms on CUDA)
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
