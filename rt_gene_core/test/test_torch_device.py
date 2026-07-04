import pytest

from rt_gene_core.torch_device import choose_auto_device, resolve_torch_device


def test_auto_prefers_mps():
    assert choose_auto_device(mps_available=True, cuda_available=True) == "mps"


def test_auto_uses_cuda_before_cpu():
    assert choose_auto_device(mps_available=False, cuda_available=True) == "cuda"


def test_auto_falls_back_to_cpu():
    assert choose_auto_device(mps_available=False, cuda_available=False) == "cpu"


def test_invalid_device_is_rejected():
    with pytest.raises(ValueError, match="Unsupported PyTorch device"):
        resolve_torch_device("not-a-device")
