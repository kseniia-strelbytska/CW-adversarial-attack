import torch
import numpy as np
from scipy.fftpack import idctn


def idct_2d(x):
    """
    Inverse DCT on the last two dimensions.

    Accepts a torch.Tensor (any device). Operation is performed on CPU using
    scipy.fftpack.idctn and the result is moved back to the original device.

    Args:
        x: torch.Tensor or numpy.ndarray with shape (..., H, W)

    Returns:
        torch.Tensor (if input was torch.Tensor) or numpy.ndarray (if input was numpy)
    """
    if isinstance(x, torch.Tensor):
        device = x.device
        np_x = x.detach().cpu().numpy()
        # apply inverse DCT on the last two axes (height, width)
        out = idctn(np_x, axes=(-2, -1), norm='ortho')
        return torch.from_numpy(out).to(device)
    else:
        return idctn(x, axes=(-2, -1), norm='ortho')
