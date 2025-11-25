from __future__ import annotations
from typing import List, Optional, Sequence, Dict, Any
from pydantic import BaseModel, Field

# --- Class Node & Edge ---
class Node(BaseModel):
    id: str
    # Đổi từ Literal sang str để tránh lỗi Pylance khi nhận dữ liệu từ DB
    type: str 
    title: Optional[str] = None

class Edge(BaseModel):
    source_id: str
    target_id: str
    # Đổi từ Literal sang str để tránh lỗi Pylance
    relation: str 

# --- Class Chunk ---
class Chunk(BaseModel):
    id: str
    document_id: str
    node_id: Optional[str] = None
    text: str
    embedding: Optional[List[float]] = None
    order: Optional[int] = None
    # Field này giúp sửa lỗi 'no attribute metadata'
    metadata: Dict[str, Any] = Field(default_factory=dict)

class RetrievalResult(BaseModel):
    chunk: Chunk
    score: float
    related_nodes: Optional[Sequence[Node]] = None
    related_edges: Optional[Sequence[Edge]] = None
