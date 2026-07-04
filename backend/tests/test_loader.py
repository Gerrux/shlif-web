from app.pipeline import loader

def test_config_loads():
    cfg = loader.get_config()
    assert cfg.rule.talc_threshold is not None

def test_classifier_absent_returns_none(monkeypatch, tmp_path):
    monkeypatch.setattr(loader.settings, "models_dir", tmp_path)
    loader.load_classifier.cache_clear()
    assert loader.load_classifier() is None
    # don't leak the tmp_path-cached `None` into later tests: clear now, while
    # models_dir is still monkeypatched, so the cache is empty (not stale) once
    # monkeypatch reverts settings.models_dir at teardown.
    loader.load_classifier.cache_clear()

def test_gpu_false_without_torch():
    assert loader.gpu_available() is False

def test_model_status_shape():
    s = loader.model_status()
    assert set(s) == {"classifier", "unet_ore", "unet_talc"}

def test_talc_unet_absent_returns_none(monkeypatch, tmp_path):
    monkeypatch.setattr(loader.settings, "models_dir", tmp_path)
    loader.load_talc_unet.cache_clear()
    assert loader.load_talc_unet() is None
    # same reasoning as test_classifier_absent_returns_none: clear now, while
    # models_dir is still monkeypatched, so no stale None survives teardown.
    loader.load_talc_unet.cache_clear()
