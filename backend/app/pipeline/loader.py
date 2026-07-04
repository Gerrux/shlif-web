from __future__ import annotations
import os, pickle
from functools import lru_cache
from app.core.settings import settings
from app.shlif import load_config
from app.shlif.config import Config

@lru_cache(maxsize=1)
def get_config() -> Config:
    return load_config()

@lru_cache(maxsize=1)
def load_classifier():
    p = settings.models_dir / "classifier.pkl"
    if not p.exists():
        return None
    m = pickle.load(open(p, "rb"))
    return m["clf"], list(m["feature_names"]), [str(c) for c in m["classes"]]

def gpu_available() -> bool:
    if os.environ.get("SHLIF_FORCE_CPU") == "1":
        return False
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False

def model_status() -> dict:
    md = settings.models_dir
    return {
        "classifier": (md / "classifier.pkl").exists(),
        "unet_ore": (md / "unet_ore.pt").exists(),
        "unet_talc": (md / "unet_talc.pt").exists(),
    }
