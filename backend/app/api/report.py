from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from app.core import paths
from app.pipeline.report import build_report_pdf
from app.runtime import get_runtime

router = APIRouter()


@router.get("/report/{jid}.pdf")
def get_report(jid: str):
    job = get_runtime().store.get(jid)
    if job is None or job.status != "done" or not job.result:
        raise HTTPException(404, "report not available")
    img = paths.images_dir() / f"{jid}.jpg"
    pdf = build_report_pdf(jid, job.mode, job.result, img if img.exists() else None)
    return Response(pdf, media_type="application/pdf",
                    headers={"Content-Disposition": f'inline; filename="shlif-{jid}.pdf"'})
