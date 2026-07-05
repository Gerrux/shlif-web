from __future__ import annotations
import io, numpy as np
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Form
from PIL import Image
from app.pipeline import closeup, panorama, loader, masks, detect
from app.core import paths
from app.runtime import get_runtime

router = APIRouter()
Image.MAX_IMAGE_PIXELS = None

@router.post("/analyze")
async def analyze(image: UploadFile = File(...), batch_id: str | None = Form(None)):
    data = await image.read()
    cfg = loader.get_config()
    iw, ih = Image.open(io.BytesIO(data)).size
    mode = detect.detect_mode(iw, ih, cfg)
    jid = get_runtime().store.create(mode, batch_id=batch_id, filename=image.filename)
    up = paths.uploads_dir() / f"{jid}_{Path(image.filename or 'up').name}"
    up.write_bytes(data)

    def work(report):
        if mode == "panorama":
            return panorama.analyze_panorama(str(up), cfg, jid, on_progress=report)
        report(0.05, "загрузка изображения")
        im = Image.open(io.BytesIO(data)).convert("RGB")
        im.thumbnail((masks.EDIT_MAX_SIDE, masks.EDIT_MAX_SIDE))
        rgb = np.asarray(im)
        r = closeup.analyze_closeup(rgb, cfg, on_progress=report)
        report(0.95, "сохранение результатов")
        disp = paths.images_dir() / f"{jid}.jpg"
        Image.fromarray(rgb).save(disp, "JPEG", quality=90)
        masks.persist_editor_artifacts(jid, r)
        h, w = rgb.shape[:2]
        return {"mode": "closeup", "verdict": r["verdict"], "sort": r["sort"],
                "text": r["text"], "size": [w, h],
                "low_conf_zones": r["low_conf_zones"]}

    get_runtime().runner.submit(jid, work)
    return {"job_id": jid}
