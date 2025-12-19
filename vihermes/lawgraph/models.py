from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel


class Node(BaseModel):
    id: str
    type: Literal[
        "Document",
        "Law",
        "Decree",
        "Circular",
        "Decision",
        "Resolution",
        "Article",
        "Clause",
        "Paragraph",
        "Term",
        "Concept",
        "CaseExample",
        "Agency",
        "ImportBatch",
        "ChangeLog",
    ]
    title: Optional[str] = None


class Edge(BaseModel):
    source_id: str
    target_id: str
    relation: Literal[
        "HAS_ARTICLE",
        "HAS_CLAUSE",
        "ISSUED_BY",
        "AMENDS",
        "REPEALS",
        "SUPPLEMENTS",
        "CITES",
        "DEFINED_IN",
        "MENTIONED_IN",
        "VERSION_OF",
        "HAS_KEYWORD",
        "APPLIES_TO",
        "REFERENCED_BY",
        "GUIDES",
        "REFERENCES",
        "BASED_ON",
        "INTERPRETS",
    ]

