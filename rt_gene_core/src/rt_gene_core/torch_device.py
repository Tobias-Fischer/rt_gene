import torch


def choose_auto_device(mps_available, cuda_available):
    if mps_available:
        return "mps"
    if cuda_available:
        return "cuda"
    return "cpu"


def resolve_torch_device(requested="auto"):
    requested = str(requested or "auto").lower()
    if requested == "auto":
        requested = choose_auto_device(
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available(),
            torch.cuda.is_available(),
        )

    if requested == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise RuntimeError("PyTorch MPS was requested but is not available")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("PyTorch CUDA was requested but is not available")
    if requested != "cpu" and requested != "mps" and not requested.startswith("cuda"):
        raise ValueError(f"Unsupported PyTorch device: {requested}")

    return torch.device(requested)
