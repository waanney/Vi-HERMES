from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class GraphTraceStep(BaseModel):
    description: str


class AnswerResponse(BaseModel):
    answer: str
    sources: List[dict]
    graph_trace: List[str] = Field(default_factory=list)
