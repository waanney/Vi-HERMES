from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel


class DocumentMetadata(BaseModel):
    document_id: str
    issuing_authority: Optional[str] = None
    effect_date: Optional[str] = None
    field: Optional[str] = None
    status: Optional[Literal["effective", "expired", "amended", "draft"]] = None
    version: Optional[str] = None
    source_url: Optional[str] = None

