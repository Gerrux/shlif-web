from __future__ import annotations
from typing import Any, Literal, Optional
from pydantic import BaseModel

Status = Literal["queued", "running", "done", "error"]

class JobRecord(BaseModel):
    id: str
    mode: str
    status: Status = "queued"
    progress: float = 0.0
    message: Optional[str] = None
    result: Optional[dict[str, Any]] = None
