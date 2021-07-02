import torch
import torch.nn.functional as F

from typing import Any, List, Optional

def _size_repr(key: str, item: Any) -> str:
    """String containing the size / shape of an object (e.g. a tensor, array)."""
    if isinstance(item, torch.Tensor) and item.dim() == 0:
        out = item.item()
    elif isinstance(item, torch.Tensor):
        out = str(list(item.size()))
    elif isinstance(item, list):
        out = f"[{len(item)}]"
    else:
        out = str(item)

    return f"{key}={out}"

def diff(x, dim: int = -1):
    """Inverse of x.cumsum(dim=dim).
    Compute differences between subsequent elements of the tensor.
    Args:
        x: Input tensor of arbitrary shape.
        dim: Dimension over which to compute the difference, {-2, -1}.
    Returns:
        diff: Tensor of the the same shape as x.
    """
    if dim == -1:
        return x - F.pad(x, (1, 0))[..., :-1]
    elif dim == -2:
        return x - F.pad(x, (0, 0, 1, 0))[..., :-1, :]
    else:
        raise ValueError("dim must be equal to -1 or -2")


def clamp_preserve_gradients(x: torch.Tensor, min: float, max: float) -> torch.Tensor:
    """Clamp the tensor while preserving gradients in the clamped region."""
    return x + (x.clamp(min, max) - x).detach()


def pad_sequence(
        sequences: List[torch.Tensor],
        padding_value: float = 0,
        max_len: Optional[int] = None,
):
    r"""Pad a list of variable length Tensors with ``padding_value``"""
    dtype = sequences[0].dtype
    device = sequences[0].device
    seq_shape = sequences[0].shape
    trailing_dims = seq_shape[1:]
    if max_len is None:
        max_len = max([s.size(0) for s in sequences])
    out_dims = (len(sequences), max_len) + trailing_dims

    out_tensor = torch.empty(*out_dims, dtype=dtype, device=device).fill_(padding_value)
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        # use index notation to prevent duplicate references to the tensor
        out_tensor[i, :length, ...] = tensor

    return out_tensor


