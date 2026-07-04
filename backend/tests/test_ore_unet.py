"""build_ore_unet guards on the checkpoint existing (and on torch/smp being
importable) so panorama.py can fall back to the classical segmenter cleanly
when neither is available -- exactly the case in this dev sandbox today."""
import numpy as np
import pytest

from app.shlif.ore_unet import build_ore_unet


def test_build_ore_unet_missing_checkpoint_returns_none(tmp_path):
    missing = tmp_path / "does_not_exist.pt"
    assert build_ore_unet(str(missing)) is None


def test_build_ore_unet_returns_none_without_torch_or_smp(tmp_path):
    # A checkpoint path that exists but isn't a real torch state dict still
    # must degrade to None, not raise -- covers "file present, torch/smp
    # absent or load fails" without needing real weights in the test.
    fake_ckpt = tmp_path / "unet_ore.pt"
    fake_ckpt.write_bytes(b"not a real checkpoint")
    assert build_ore_unet(str(fake_ckpt)) is None


class _CountingOreModel:
    """Stand-in for the real U-Net: records the batch size of every forward
    call. Flags class 1 (ore) per-sample based on that sample's own mean
    value (positive vs. negative after ImageNet normalisation) -- content
    dependent, independent of what else shares the batch, exactly how a real
    batched CNN behaves (each sample is processed independently of its
    batch-mates)."""
    def __init__(self):
        self.batch_sizes = []

    def __call__(self, batch):
        import torch
        self.batch_sizes.append(batch.shape[0])
        n, c, h, w = batch.shape
        ore = (batch.mean(dim=(1, 2, 3)) > 0).float().view(n, 1, 1)
        out = torch.zeros(n, 2, h, w)
        out[:, 1] = ore
        out[:, 0] = 1 - ore
        return out


def _quadrant_tile(bright_row, bright_col):
    """1024x1024 tile split into four 512x512 quadrants; (bright_row,
    bright_col) in {0,1}x{0,1} is filled 240 (bright), the rest 10 (dark)."""
    rgb = np.full((1024, 1024, 3), 10, np.uint8)
    rgb[bright_row * 512:(bright_row + 1) * 512,
        bright_col * 512:(bright_col + 1) * 512] = 240
    return rgb


@pytest.fixture(autouse=True)
def _no_clahe(monkeypatch):
    # CLAHE's behaviour on a perfectly flat crop isn't the point of this
    # test -- stub it to identity so only the batching logic is exercised.
    monkeypatch.setattr("app.shlif.preprocess.wb_clahe", lambda rgb, *a, **k: rgb)


def test_batches_all_crops_into_one_forward_pass_when_within_batch_size():
    pytest.importorskip("torch")
    from app.shlif.ore_unet import ore_unet_mask

    rgb = _quadrant_tile(0, 1)  # top-right quadrant bright
    model = _CountingOreModel()

    mask = ore_unet_mask(rgb, model, "cpu", tile=512, batch_size=32)

    assert model.batch_sizes == [4], model.batch_sizes
    assert mask[0:512, 512:1024].all()
    assert not mask[0:512, 0:512].any()
    assert not mask[512:1024, :].any()


def test_chunks_when_more_crops_than_batch_size():
    pytest.importorskip("torch")
    from app.shlif.ore_unet import ore_unet_mask

    rgb = _quadrant_tile(1, 0)  # bottom-left quadrant bright
    model = _CountingOreModel()

    mask = ore_unet_mask(rgb, model, "cpu", tile=512, batch_size=2)

    assert model.batch_sizes == [2, 2], model.batch_sizes
    assert mask[512:1024, 0:512].all()
    assert not mask[0:512, :].any()
    assert not mask[512:1024, 512:1024].any()


def test_use_amp_true_for_cuda_devices():
    from app.shlif.ore_unet import _use_amp
    assert _use_amp("cuda") is True
    assert _use_amp("cuda:0") is True


def test_use_amp_false_for_cpu():
    from app.shlif.ore_unet import _use_amp
    assert _use_amp("cpu") is False
