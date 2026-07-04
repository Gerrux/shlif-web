from __future__ import annotations
from pathlib import Path
from .settings import settings

def _ensure(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def uploads_dir() -> Path: return _ensure(settings.data_dir / "uploads")
def images_dir() -> Path: return _ensure(settings.data_dir / "images")
def masks_dir(job_id: str) -> Path: return _ensure(settings.data_dir / "masks" / job_id)
def maps_dir(job_id: str) -> Path: return _ensure(settings.data_dir / "maps" / job_id)
def db_path() -> Path:
    _ensure(settings.data_dir)
    return settings.data_dir / "shlif.db"
