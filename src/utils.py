import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    #To make experiments reproducible
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # If GPU is used, this helps make things more repeatable.
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)