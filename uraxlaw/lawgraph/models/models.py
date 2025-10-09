from __future__ import annotations
"""Centralized Pydantic / dataclass models for lawgraph.

This file was created to host all schema models (Relation, Clause, Point, Article, LawSchema)
plus the QuerySlots dataclass used for query slot extraction / routing.

NOTE: Original definitions remain in their previous module; callers can begin migrating
imports to `uraxlaw.lawgraph.models.models` (or `uraxlaw.lawgraph.models`) without
any functional change.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

from dateutil.parser import parse as parse_date
from pydantic import BaseModel, Field, field_validator

# =======================
# Pydantic Graph Models
# =======================
class Relation(BaseModel):
    """Legal relationship between provisions/documents.

    type: MODIFIES | ADDS | REPEALS | REPLACES | REFERS_TO | DEFINES
    source / target: free-form textual identifiers or provision references.
    Dates are normalized to ISO (yyyy-mm-dd) when parseable.
    """
    type: str
    source: str
    target: str
    effective_date: Optional[str] = None
    expiry_date: Optional[str] = None
    description: Optional[str] = None

    @field_validator("effective_date", "expiry_date")
    @classmethod
    def _norm_date(cls, v: Optional[str]):  # noqa: D401
        if not v:
            return None
        try:
            return parse_date(v, dayfirst=True).date().isoformat()
        except Exception:  # pragma: no cover - permissive fallback
            return v


class Clause(BaseModel):
    clause_number: int
    content: str
    effective_date: Optional[str] = None
    expiry_date: Optional[str] = None
    relations: List[Relation] = Field(default_factory=list)
    points: List["Point"] = Field(default_factory=list)

    @field_validator("effective_date", "expiry_date")
    @classmethod
    def _norm_date(cls, v: Optional[str]):
        if not v:
            return None
        try:
            return parse_date(v, dayfirst=True).date().isoformat()
        except Exception:  # pragma: no cover
            return v


class Point(BaseModel):
    point_symbol: str  # e.g. "a", "b", "c" (Điểm a, b, c...)
    content: str
    effective_date: Optional[str] = None
    expiry_date: Optional[str] = None
    relations: List[Relation] = Field(default_factory=list)

    @field_validator("effective_date", "expiry_date")
    @classmethod
    def _norm_date(cls, v: Optional[str]):
        if not v:
            return None
        try:
            return parse_date(v, dayfirst=True).date().isoformat()
        except Exception:  # pragma: no cover
            return v


# Resolve forward refs between Clause <-> Point
Clause.model_rebuild()
Point.model_rebuild()


class Article(BaseModel):
    article_number: int
    title: Optional[str] = None
    content: Optional[str] = None
    effective_date: Optional[str] = None
    expiry_date: Optional[str] = None
    clauses: List[Clause] = Field(default_factory=list)

    @field_validator("effective_date", "expiry_date")
    @classmethod
    def _norm_date(cls, v: Optional[str]):
        if not v:
            return None
        try:
            return parse_date(v, dayfirst=True).date().isoformat()
        except Exception:  # pragma: no cover
            return v


class LawSchema(BaseModel):
    law_name: str
    law_id: Optional[str] = None
    document_type: Optional[str] = None  # Luật, Nghị định, Thông tư...
    issued_by: Optional[str] = None
    signer: Optional[str] = None
    issued_date: Optional[str] = None
    promulgation_date: Optional[str] = None
    effective_date: Optional[str] = None
    expiry_date: Optional[str] = None
    scope: Optional[str] = None  # Phạm vi điều chỉnh
    language: str = "vi"
    aliases: List[str] = Field(default_factory=list)
    modified_by: List[str] = Field(default_factory=list)
    articles: List[Article] = Field(default_factory=list)

    @field_validator("issued_date", "promulgation_date", "effective_date", "expiry_date")
    @classmethod
    def _norm_date(cls, v: Optional[str]):
        if not v:
            return None
        try:
            return parse_date(v, dayfirst=True).date().isoformat()
        except Exception:  # pragma: no cover
            return v


# =======================
# Query Slot Dataclass
# =======================
@dataclass
class QuerySlots:
    """Captures structured hints parsed from a natural-language legal query."""
    law_ids: List[str] = field(default_factory=list)
    law_names: List[str] = field(default_factory=list)
    article_numbers: List[int] = field(default_factory=list)
    clause_numbers: List[int] = field(default_factory=list)
    as_of: Optional[str] = None
    rel_types: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    open_text: str = ""
    is_ambiguous: bool = False


__all__ = [
    "Relation",
    "Clause",
    "Point",
    "Article",
    "LawSchema",
    "QuerySlots",
]

