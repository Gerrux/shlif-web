from __future__ import annotations
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from app.core import paths

router = APIRouter()

@router.get("/tiles/{jid}/manifest.json")
def get_manifest(jid: str):
    p = paths.tiles_dir(jid) / "manifest.json"
    if not p.exists():
        raise HTTPException(404, "tile pyramid not found")
    return FileResponse(p, media_type="application/json")

@router.get("/tiles/{jid}/{level}/{col}_{row}.jpg")
def get_tile(jid: str, level: int, col: int, row: int):
    p = paths.tiles_dir(jid) / str(level) / f"{col}_{row}.jpg"
    if not p.exists():
        raise HTTPException(404, "tile not found")
    return FileResponse(p, media_type="image/jpeg")
