import torch
from typing import Optional, Union


def safe_to_device(
    data: Union[torch.Tensor, dict, list],
    device: torch.device,
) -> Union[torch.Tensor, dict, list]:
    if isinstance(data, dict):
        return {k: safe_to_device(v, device) for k, v in data.items()}
    if isinstance(data, list):
        return [safe_to_device(v, device) for v in data]
    if hasattr(data, "to"):
        return data.to(device)
    return data


def detach_to_cpu(tensor: torch.Tensor) -> torch.Tensor:
    if tensor is None:
        return tensor
    return tensor.detach().cpu()


def clamp_nan_inf(tensor: torch.Tensor, replace: float = 0.0) -> torch.Tensor:
    if tensor is None:
        return tensor
    result = tensor.clone()
    mask = torch.isnan(result) | torch.isinf(result)
    if mask.any():
        result[mask] = replace
    return result


def safe_mean(tensor: torch.Tensor, dim: Optional[int] = None) -> torch.Tensor:
    if tensor is None or tensor.numel() == 0:
        return torch.tensor(0.0)
    cleaned = clamp_nan_inf(tensor)
    return cleaned.mean(dim=dim)


def ensure_float_tensor(
    value: Union[torch.Tensor, float, int],
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        t = value.float()
    else:
        t = torch.tensor(float(value), dtype=torch.float32)
    if device is not None:
        t = t.to(device)
    return t
