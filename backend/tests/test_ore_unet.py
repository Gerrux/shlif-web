"""build_ore_unet guards on the checkpoint existing (and on torch/smp being
importable) so panorama.py can fall back to the classical segmenter cleanly
when neither is available -- exactly the case in this dev sandbox today."""
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
