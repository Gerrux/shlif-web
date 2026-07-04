from __future__ import annotations
from fastapi import APIRouter, HTTPException
from app.runtime import get_runtime

router = APIRouter()

@router.get("/jobs/{jid}")
def get_job(jid: str):
    rec = get_runtime().store.get(jid)
    if rec is None:
        raise HTTPException(404, "job not found")
    return rec
