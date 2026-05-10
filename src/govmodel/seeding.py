"""Reproducibility helper.

Set every RNG we know about, mark cuDNN deterministic when possible.
Returns the seed actually used so it can be logged into training metadata.
"""
from __future__ import annotations

import logging
import os
import random

logger = logging.getLogger(__name__)


def seed_everything(seed: int = 42, *, deterministic_cudnn: bool = True) -> int:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_cudnn:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    logger.info("seeded RNGs", extra={"seed": seed, "cudnn_deterministic": deterministic_cudnn})
    return seed
