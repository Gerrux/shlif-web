from __future__ import annotations
import numpy as np
from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from app.core import paths
from app.pipeline import masks as M, loader
from app.runtime import get_runtime

router = APIRouter()

@router.get("/masks/{jid}/{layer}.png")
def get_mask(jid: str, layer: str):
    p = paths.masks_dir(jid) / f"{layer}.png"
    if layer not in {"phases", "talc"} or not p.exists():
        raise HTTPException(404, "mask not found")
    return FileResponse(p, media_type="image/png")

@router.get("/maps/{jid}/{name}.png")
def get_map(jid: str, name: str):
    p = paths.maps_dir(jid) / f"{name}.png"
    if name not in {"superpixels", "darkness", "confidence"} or not p.exists():
        raise HTTPException(404, "map not found")
    return FileResponse(p, media_type="image/png")

@router.get("/images/{jid}.jpg")
def get_image(jid: str):
    p = paths.images_dir() / f"{jid}.jpg"
    if not p.exists():
        raise HTTPException(404, "image not found")
    return FileResponse(p, media_type="image/jpeg")

@router.post("/masks/{jid}")
async def save_masks(jid: str, phases: UploadFile = File(...), talc: UploadFile = File(...)):
    pm = M.decode_png_gray(await phases.read()).astype(np.uint8)
    tk = M.decode_png_gray(await talc.read()) > 127
    paths.masks_dir(jid).joinpath("phases.png").write_bytes(M.encode_png_gray(pm))
    paths.masks_dir(jid).joinpath("talc.png").write_bytes(M.encode_png_gray(tk.astype(np.uint8) * 255))
    su, mg, mx = M.split_phase_map(pm)
    cfg = loader.get_config()
    v = M.verdict_from_masks_dict(su, mg, mx, tk & mx, cfg)
    get_runtime().store.log_correction(jid, "phases+talc", int(pm.size))
    return v
