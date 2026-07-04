"""talc_unet_mask/build_talc_unet gate fp16 autocast + TF32 on CUDA only; the
CPU fallback path must stay plain fp32. This only covers the device-gating
logic -- there's no live CUDA in this sandbox to exercise the actual
autocast/TF32 behaviour; that needs the real L4 VM."""
from app.shlif.talc_unet import _use_amp


def test_use_amp_true_for_cuda_devices():
    assert _use_amp("cuda") is True
    assert _use_amp("cuda:0") is True


def test_use_amp_false_for_cpu():
    assert _use_amp("cpu") is False
