from app.pipeline import loader

def test_config_loads():
    cfg = loader.get_config()
    assert cfg.rule.talc_threshold is not None

def test_classifier_absent_returns_none(monkeypatch, tmp_path):
    monkeypatch.setattr(loader.settings, "models_dir", tmp_path)
    loader.load_classifier.cache_clear()
    assert loader.load_classifier() is None

def test_gpu_false_without_torch():
    assert loader.gpu_available() is False

def test_model_status_shape():
    s = loader.model_status()
    assert set(s) == {"classifier", "unet_ore", "unet_talc"}
