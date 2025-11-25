from __future__ import annotations
from typing import Optional
from pydantic import BaseModel

class DocumentMetadata(BaseModel):
    """Lưu trữ thông tin metadata của tài liệu sau khi parse."""
    title: str
    document_id: str
    text: str
    issuing_authority: Optional[str] = None
    effect_date: Optional[str] = None
    field: Optional[str] = None
    # Sửa status thành str để linh hoạt hơn, tránh lỗi validate
    status: Optional[str] = None
