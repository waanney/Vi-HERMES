"""lawgraph.models package: re-exports all model classes.

Fixes previous empty __init__ that caused ImportError when doing:
    from uraxlaw.lawgraph.models import Relation, LawSchema, ...
"""
from .models import Relation, Clause, Point, Article, LawSchema, QuerySlots

__all__ = [
    "Relation",
    "Clause",
    "Point",
    "Article",
    "LawSchema",
    "QuerySlots",
]

