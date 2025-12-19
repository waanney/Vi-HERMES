from __future__ import annotations

from typing import List, Optional, Sequence

from pydantic import BaseModel

from vihermes.lawgraph.models import Edge, Node


class Chunk(BaseModel):
    id: str
    document_id: str
    node_id: Optional[str] = None
    text: str
    embedding: Optional[List[float]] = None
    order: Optional[int] = None


class RetrievalResult(BaseModel):
    chunk: Chunk
    score: float
    related_nodes: Optional[Sequence[Node]] = None
    related_edges: Optional[Sequence[Edge]] = None

