from __future__ import annotations
from pathlib import Path

class Settings:
    """Runtime paths + limits. Env override via SHLIF_DATA_DIR / SHLIF_MODELS_DIR."""
    def __init__(self) -> None:
        import os
        root = Path(__file__).resolve().parents[2]  # backend/
        self.data_dir = Path(os.environ.get("SHLIF_DATA_DIR", root / "data"))
        self.models_dir = Path(os.environ.get("SHLIF_MODELS_DIR", root / "models"))
        self.max_display_px = int(os.environ.get("SHLIF_MAX_DISPLAY_PX", 4_000_000))

settings = Settings()
