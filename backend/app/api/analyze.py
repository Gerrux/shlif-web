from __future__ import annotations
import io, numpy as np
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Form
from PIL import Image
from app.pipeline import closeup, panorama, loader, masks
from app.core import paths
from app.runtime import get_runtime

router = APIRouter()
Image.MAX_IMAGE_PIXELS = None

def _persist_maps(jid, r):
    md = paths.masks_dir(jid); mp = paths.maps_dir(jid)
    (md / "phases.png").write_bytes(masks.encode_png_gray(r["phase_map"]))
    (md / "talc.png").write_bytes(masks.encode_png_gray((r["talc"].astype(np.uint8) * 255)))
    (mp / "superpixels.png").write_bytes(masks.encode_png_u16(r["superpixels"]))
    (mp / "darkness.png").write_bytes(masks.encode_png_gray(r["darkness"]))

@router.post("/analyze")
async def analyze(image: UploadFile = File(...), mode: str = Form("closeup")):
    data = await image.read()
    jid = get_runtime().store.create(mode)
    up = paths.uploads_dir() / f"{jid}_{Path(image.filename or 'up').name}"
    up.write_bytes(data)

    def work():
        cfg = loader.get_config()
        im = Image.open(io.BytesIO(data)).convert("RGB")
        if mode == "panorama":
            return panorama.analyze_panorama(str(up), cfg, jid)
        im.thumbnail((2400, 2400))
        rgb = np.asarray(im)
        r = closeup.analyze_closeup(rgb, cfg)
        # save display image + editor layers/maps
        disp = paths.images_dir() / f"{jid}.jpg"
        Image.fromarray(rgb).save(disp, "JPEG", quality=90)
        _persist_maps(jid, r)
        h, w = rgb.shape[:2]
        return {"mode": "closeup", "verdict": r["verdict"], "sort": r["sort"],
                "text": r["text"], "size": [w, h]}

    get_runtime().runner.submit(jid, work)
    return {"job_id": jid}
