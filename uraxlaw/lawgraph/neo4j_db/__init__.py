"""Neo4j query package for lawgraph.

Re-exports Cypher query constants defined in queries.py so consumers can do:
    from uraxlaw.lawgraph.neo4j_db import SCHEMA_CYPHER, INGEST_CYPHER, SEARCH_TOP_ARTICLES, ...
"""
from .queries import (
    SCHEMA_CYPHER,
    INGEST_CYPHER,
    GET_ARTICLE_CONTENT,
    GET_MODIFIERS,
    SEARCH_TOP_ARTICLES,
    SEARCH_ALL_NODES,
    NODE_CONTEXT,
    STRUCTURED_LAW_QUERY,
    STRUCTURED_ARTICLE_QUERY,
    STRUCTURED_CLAUSE_QUERY,
)

__all__ = [
    "SCHEMA_CYPHER",
    "INGEST_CYPHER",
    "GET_ARTICLE_CONTENT",
    "GET_MODIFIERS",
    "SEARCH_TOP_ARTICLES",
    "SEARCH_ALL_NODES",
    "NODE_CONTEXT",
    "STRUCTURED_LAW_QUERY",
    "STRUCTURED_ARTICLE_QUERY",
    "STRUCTURED_CLAUSE_QUERY",
]

