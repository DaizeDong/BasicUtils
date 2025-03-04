import math
import numbers
from typing import Iterable

import numpy as np
import torch


def all_nan(input, strict=False):
    """Recursively check if all elements in the input are NaN."""
    if input is None:  # Treat None as NaN
        return True
    elif isinstance(input, (bool, str)):
        return False
    elif isinstance(input, numbers.Number):  # int, float, complex
        return math.isnan(input) if isinstance(input, float) else False
    elif isinstance(input, np.ndarray):
        return np.all(np.isnan(input))
    elif isinstance(input, torch.Tensor):
        return input.isnan().all()
    elif isinstance(input, dict):
        return all(all_nan(value) for value in input.values())
    elif isinstance(input, Iterable):
        return all(all_nan(value) for value in input)
    else:
        if strict:
            raise TypeError(f"Unsupported input type: {type(input)}")
        return False
