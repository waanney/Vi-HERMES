from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

from neo4j import GraphDatabase

from uraxlaw.lawgraph.models import Edge, Node


# Node labels
NODE_LABELS = [
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

# Relationship types
REL_TYPES = [
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


class Neo4jClient:
    def __init__(self, uri: str, user: str, password: str) -> None:
        self._driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self) -> None:
        self._driver.close()

    def init_schema(self) -> None:
        """Initialize Neo4j schema: constraints, indexes, and full-text search."""
        with self._driver.session() as session:
            # 1. Unique constraints for identifiers
            constraints = [
                "CREATE CONSTRAINT document_doc_id_unique IF NOT EXISTS FOR (d:Document) REQUIRE d.doc_id IS UNIQUE",
                "CREATE CONSTRAINT article_article_id_unique IF NOT EXISTS FOR (a:Article) REQUIRE a.article_id IS UNIQUE",
                "CREATE CONSTRAINT clause_clause_id_unique IF NOT EXISTS FOR (c:Clause) REQUIRE c.clause_id IS UNIQUE",
                "CREATE CONSTRAINT term_name_unique IF NOT EXISTS FOR (t:Term) REQUIRE t.normalized_name IS UNIQUE",
                "CREATE CONSTRAINT agency_name_unique IF NOT EXISTS FOR (a:Agency) REQUIRE a.name IS UNIQUE",
                "CREATE CONSTRAINT import_batch_id_unique IF NOT EXISTS FOR (b:ImportBatch) REQUIRE b.batch_id IS UNIQUE",
            ]
            for constraint in constraints:
                try:
                    session.run(constraint)
                except Exception:
                    pass  # Constraint may already exist

            # 2. Indexes for common queries
            indexes = [
                "CREATE INDEX document_title_idx IF NOT EXISTS FOR (d:Document) ON (d.title)",
                "CREATE INDEX document_number_idx IF NOT EXISTS FOR (d:Document) ON (d.number)",
                "CREATE INDEX document_status_idx IF NOT EXISTS FOR (d:Document) ON (d.status)",
                "CREATE INDEX document_type_idx IF NOT EXISTS FOR (d:Document) ON (d.doc_type)",
                "CREATE INDEX article_number_idx IF NOT EXISTS FOR (a:Article) ON (a.number)",
                "CREATE INDEX clause_number_idx IF NOT EXISTS FOR (c:Clause) ON (c.number)",
                "CREATE INDEX term_name_idx IF NOT EXISTS FOR (t:Term) ON (t.normalized_name)",
                "CREATE INDEX agency_name_idx IF NOT EXISTS FOR (a:Agency) ON (a.name)",
            ]
            for index in indexes:
                try:
                    session.run(index)
                except Exception:
                    pass  # Index may already exist

            # 3. Full-text search index
            try:
                session.run("""
                    CALL db.index.fulltext.createNodeIndex(
                        "lawTextIndex",
                        ["Document", "Article", "Clause", "Term"],
                        ["raw_text", "text", "name"]
                    )
                """)
            except Exception:
                pass  # Index may already exist

    def run_cypher(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        params = params or {}
        with self._driver.session() as session:
            return session.run(query, params).data()

    def upsert_node(self, label: str, node_id: str, properties: Optional[Dict[str, Any]] = None) -> None:
        """Upsert a node with appropriate identifier property."""
        properties = properties or {}
        
        # Determine identifier property based on node label
        if label == "Document":
            id_prop = "doc_id"
        elif label == "Article":
            id_prop = "article_id"
        elif label == "Clause":
            id_prop = "clause_id"
        else:
            id_prop = "id"
        
        with self._driver.session() as session:
            session.run(
                f"MERGE (n:{label} {{{id_prop}:$id}}) SET n += $props",
                {"id": node_id, "props": properties},
            )

    def upsert_edge(
        self,
        src_label: str,
        src_id: str,
        relation: str,
        tgt_label: str,
        tgt_id: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Upsert an edge between two nodes."""
        if relation not in REL_TYPES:
            raise ValueError(f"Unsupported relation: {relation}. Supported: {REL_TYPES}")
        properties = properties or {}
        
        # Determine identifier property based on node type
        src_id_prop = "doc_id" if src_label == "Document" else ("article_id" if src_label == "Article" else ("clause_id" if src_label == "Clause" else "id"))
        tgt_id_prop = "doc_id" if tgt_label == "Document" else ("article_id" if tgt_label == "Article" else ("clause_id" if tgt_label == "Clause" else "id"))
        
        with self._driver.session() as session:
            session.run(
                f"""
                MERGE (s:{src_label} {{{src_id_prop}:$sid}})
                MERGE (t:{tgt_label} {{{tgt_id_prop}:$tid}})
                MERGE (s)-[r:{relation}]->(t)
                SET r += $props
                """,
                {"sid": src_id, "tid": tgt_id, "props": properties},
            )

    def expand_related(self, node_id: Optional[str], max_hops: int = 1, limit: int = 50) -> List[Tuple[Node, Edge]]:
        """Expand related nodes via relationships."""
        if not node_id:
            return []
        
        rel_types_union = "|".join(REL_TYPES)
        query = f"""
        MATCH (n)
        WHERE n.id = $id OR n.doc_id = $id OR n.article_id = $id OR n.clause_id = $id
        MATCH (n)-[r:{rel_types_union}]-(m)
        RETURN type(r) AS rel_type,
               labels(startNode(r)) AS src_labels,
               labels(m) AS tgt_labels,
               startNode(r).id AS src_id,
               startNode(r).doc_id AS src_doc_id,
               startNode(r).article_id AS src_article_id,
               startNode(r).clause_id AS src_clause_id,
               m.id AS tgt_id,
               m.doc_id AS tgt_doc_id,
               m.article_id AS tgt_article_id,
               m.clause_id AS tgt_clause_id,
               m.title AS title
        LIMIT $limit
        """
        with self._driver.session() as session:
            recs = session.run(query, {"id": node_id, "limit": max(1, limit)}).data()

        results: List[Tuple[Node, Edge]] = []
        for rec in recs:
            tgt_labels = rec.get("tgt_labels") or []
            node_type = self._primary_label(tgt_labels)
            
            # Get target node ID from appropriate property
            tgt_node_id = (
                rec.get("tgt_id")
                or rec.get("tgt_doc_id")
                or rec.get("tgt_article_id")
                or rec.get("tgt_clause_id")
            )
            
            # Get source node ID from appropriate property
            src_node_id = (
                rec.get("src_id")
                or rec.get("src_doc_id")
                or rec.get("src_article_id")
                or rec.get("src_clause_id")
            )
            
            related_node = Node(id=str(tgt_node_id), type=node_type, title=rec.get("title"))
            edge = Edge(
                source_id=str(src_node_id),
                target_id=str(tgt_node_id),
                relation=rec.get("rel_type")
            )
            results.append((related_node, edge))
        return results

    def _primary_label(self, labels: Iterable[str]) -> str:
        """Get primary label from list of labels."""
        for lbl in labels:
            if lbl in NODE_LABELS:
                return lbl
        return "Document"  # Default fallback

