"""Centralized Cypher query definitions for lawgraph Neo4j integration.

Exports all Cypher query / template constants previously embedded in test.py.
Other modules should import from `uraxlaw.lawgraph.neo4j_db` or
`uraxlaw.lawgraph.neo4j_db.queries`.
"""
from __future__ import annotations

# Schema (constraints & indexes)
SCHEMA_CYPHER = [
    "CREATE CONSTRAINT law_id_unique IF NOT EXISTS FOR (l:Law) REQUIRE l.law_id IS UNIQUE",
    "CREATE INDEX law_name_idx IF NOT EXISTS FOR (l:Law) ON (l.law_name)",
    "CREATE INDEX article_key_idx IF NOT EXISTS FOR (a:Article) ON (a.article_key)",
    "CREATE INDEX clause_key_idx IF NOT EXISTS FOR (c:Clause) ON (c.clause_key)",
    "CREATE INDEX point_key_idx IF NOT EXISTS FOR (p:Point) ON (p.point_key)",
]

# Ingestion parametric statements
INGEST_CYPHER = {
    "law": (
        "MERGE (l:Law {law_id: coalesce($law_id, $law_name)})\n"
        "SET l.law_name=$law_name, l.document_type=$document_type, l.issued_by=$issued_by, l.signer=$signer, "
        "l.issued_date=$issued_date, l.promulgation_date=$promulgation_date, l.effective_date=$effective_date, "
        "l.expiry_date=$expiry_date, l.scope=$scope, l.language=$language, l.aliases=$aliases, l.modified_by=$modified_by"
    ),
    "article": (
        "MATCH (l:Law {law_id: coalesce($law_id, $law_name)})\n"
        "MERGE (a:Article {article_key: $article_key})\n"
        "SET a.article_number=$article_number, a.title=$title, a.content=$content, a.effective_date=$effective_date, a.expiry_date=$expiry_date, a.law_id=l.law_id, a.law_name=l.law_name\n"
        "MERGE (l)-[:HAS_ARTICLE]->(a)"
    ),
    "clause": (
        "MATCH (a:Article {article_key: $article_key})\n"
        "MERGE (c:Clause {clause_key: $clause_key})\n"
        "SET c.clause_number=$clause_number, c.content=$content, c.effective_date=$effective_date, c.expiry_date=$expiry_date, c.article_key=a.article_key\n"
        "MERGE (a)-[:HAS_CLAUSE]->(c)"
    ),
    "point": (
        "MATCH (c:Clause {clause_key: $clause_key})\n"
        "MERGE (p:Point {point_key: $point_key})\n"
        "SET p.point_symbol=$point_symbol, p.point_symbol_norm=$point_symbol_norm, p.content=$content, p.effective_date=$effective_date, p.expiry_date=$expiry_date, p.clause_key=c.clause_key\n"
        "MERGE (c)-[:HAS_POINT]->(p)"
    ),
    "relation": (
        "WITH $source_point_key AS spk, $source_clause_key AS sck, $source_article_key AS sak, $source_name AS sn, "
        "$target_point_key AS tpk, $target_clause_key AS tck, $target_article_key AS tak, $target_name AS tn, "
        "$type AS tp, $effective_date AS ed, $expiry_date AS exd, $description AS ds\n"
        "OPTIONAL MATCH (src_point:Point {point_key: spk})\n"
        "OPTIONAL MATCH (src_clause:Clause {clause_key: sck})\n"
        "OPTIONAL MATCH (src_article:Article {article_key: sak})\n"
        "WITH CASE\n"
        "        WHEN src_point IS NOT NULL THEN src_point\n"
        "        WHEN src_clause IS NOT NULL THEN src_clause\n"
        "        WHEN src_article IS NOT NULL THEN src_article\n"
        "        ELSE NULL\n"
        "     END AS src_candidate, sn, tpk, tck, tak, tn, tp, ed, exd, ds\n"
        "FOREACH (_ IN CASE WHEN src_candidate IS NULL AND sn IS NOT NULL THEN [1] ELSE [] END |\n"
        "    MERGE (src_doc_merge:Document {name: sn})\n"
        ")\n"
        "WITH src_candidate, sn, tpk, tck, tak, tn, tp, ed, exd, ds\n"
        "OPTIONAL MATCH (src_doc:Document {name: sn})\n"
        "WITH coalesce(src_candidate, src_doc) AS src, sn, tpk, tck, tak, tn, tp, ed, exd, ds\n"
        "OPTIONAL MATCH (tgt_point:Point {point_key: tpk})\n"
        "OPTIONAL MATCH (tgt_clause:Clause {clause_key: tck})\n"
        "OPTIONAL MATCH (tgt_article:Article {article_key: tak})\n"
        "WITH src, sn, CASE\n"
        "        WHEN tgt_point IS NOT NULL THEN tgt_point\n"
        "        WHEN tgt_clause IS NOT NULL THEN tgt_clause\n"
        "        WHEN tgt_article IS NOT NULL THEN tgt_article\n"
        "        ELSE NULL\n"
        "     END AS tgt_candidate, tn, tp, ed, exd, ds\n"
        "FOREACH (_ IN CASE WHEN tgt_candidate IS NULL AND tn IS NOT NULL THEN [1] ELSE [] END |\n"
        "    MERGE (tgt_doc_merge:Document {name: tn})\n"
        ")\n"
        "WITH src, sn, tgt_candidate, tn, tp, ed, exd, ds\n"
        "OPTIONAL MATCH (tgt_doc:Document {name: tn})\n"
        "WITH src, sn, coalesce(tgt_candidate, tgt_doc) AS tgt, tp, ed, exd, ds\n"
        "WHERE src IS NOT NULL AND tgt IS NOT NULL\n"
        "FOREACH (_ IN CASE WHEN tp='MODIFIES' THEN [1] ELSE [] END |\n"
        "    MERGE (src)-[r:MODIFIES]->(tgt)\n"
        "    SET r.effective_date = ed, r.expiry_date = exd, r.description = ds, r.source = sn\n"
        ")\n"
        "FOREACH (_ IN CASE WHEN tp='ADDS' THEN [1] ELSE [] END |\n"
        "    MERGE (src)-[r:ADDS]->(tgt)\n"
        "    SET r.effective_date = ed, r.expiry_date = exd, r.description = ds, r.source = sn\n"
        ")\n"
        "FOREACH (_ IN CASE WHEN tp='REPEALS' THEN [1] ELSE [] END |\n"
        "    MERGE (src)-[r:REPEALS]->(tgt)\n"
        "    SET r.effective_date = ed, r.expiry_date = exd, r.description = ds, r.source = sn\n"
        ")\n"
        "FOREACH (_ IN CASE WHEN tp='REPLACES' THEN [1] ELSE [] END |\n"
        "    MERGE (src)-[r:REPLACES]->(tgt)\n"
        "    SET r.effective_date = ed, r.expiry_date = exd, r.description = ds, r.source = sn\n"
        ")\n"
        "FOREACH (_ IN CASE WHEN tp='REFERS_TO' THEN [1] ELSE [] END |\n"
        "    MERGE (src)-[r:REFERS_TO]->(tgt)\n"
        "    SET r.effective_date = ed, r.expiry_date = exd, r.description = ds, r.source = sn\n"
        ")\n"
        "FOREACH (_ IN CASE WHEN tp='DEFINES' THEN [1] ELSE [] END |\n"
        "    MERGE (src)-[r:DEFINES]->(tgt)\n"
        "    SET r.effective_date = ed, r.expiry_date = exd, r.description = ds, r.source = sn\n"
        ")"
    ),
}

# Query helpers
GET_ARTICLE_CONTENT = (
    "MATCH (l:Law {law_name:$law_name})-[:HAS_ARTICLE]->(a:Article {article_number:$article_number})\n"
    "RETURN a.title AS title, a.content AS content, l.effective_date AS law_effective"
)

GET_MODIFIERS = (
    "MATCH (l:Law {law_name:$law_name})-[:HAS_ARTICLE]->(a:Article {article_number:$article_number})\n"
    "OPTIONAL MATCH (src:Document)-[r:MODIFIES]->(a)\n"
    "WHERE src IS NOT NULL\n"
    "RETURN src.name AS source, r.effective_date AS effective, r.expiry_date AS expiry\n"
    "UNION\n"
    "MATCH (l:Law {law_name:$law_name})-[:HAS_ARTICLE]->(a:Article {article_number:$article_number})-[:HAS_CLAUSE]->(c:Clause)\n"
    "OPTIONAL MATCH (src:Document)-[r:MODIFIES]->(c)\n"
    "WHERE src IS NOT NULL\n"
    "RETURN src.name AS source, r.effective_date AS effective, r.expiry_date AS expiry\n"
    "UNION\n"
    "MATCH (l:Law {law_name:$law_name})-[:HAS_ARTICLE]->(a:Article {article_number:$article_number})-[:HAS_CLAUSE]->(c:Clause)-[:HAS_POINT]->(p:Point)\n"
    "OPTIONAL MATCH (src:Document)-[r:MODIFIES]->(p)\n"
    "WHERE src IS NOT NULL\n"
    "RETURN src.name AS source, r.effective_date AS effective, r.expiry_date AS expiry"
)

SEARCH_TOP_ARTICLES = (
    "MATCH (l:Law)-[:HAS_ARTICLE]->(a:Article)\n"
    "WITH l, a,\n"
    "     CASE WHEN size($article_numbers) > 0 AND a.article_number IN $article_numbers THEN 2 ELSE 0 END AS score_article,\n"
    "     CASE WHEN size($law_ids) > 0 AND toUpper(coalesce(l.law_id,'')) IN $law_ids THEN 2 ELSE 0 END AS score_law_id,\n"
    "     CASE WHEN $law_name_term <> '' AND toLower(coalesce(l.law_name,'')) CONTAINS $law_name_term THEN 1 ELSE 0 END AS score_law_name,\n"
    "     CASE WHEN $term <> '' AND (toLower(coalesce(a.title,'')) CONTAINS $term OR toLower(coalesce(a.content,'')) CONTAINS $term) THEN 1 ELSE 0 END AS score_text\n"
    "WITH l, a, (score_article + score_law_id + score_law_name + score_text) AS score\n"
    "WHERE score > 0 OR $term = ''\n"
    "OPTIONAL MATCH (doc:Document)-[rel:MODIFIES|ADDS|REPEALS|REPLACES|REFERS_TO|DEFINES]->(a)\n"
    "WITH l, a, score, collect({source: doc.name, type: rel.type, effective_date: rel.effective_date, expiry_date: rel.expiry_date, description: rel.description}) AS rels\n"
    "RETURN l.law_name AS law_name, l.law_id AS law_id, a.article_number AS article_number, a.title AS title, a.content AS content, score, rels\n"
    "ORDER BY score DESC, a.article_number ASC\n"
    "LIMIT $limit"
)

SEARCH_ALL_NODES = (
    "MATCH (n)\n"
    "WITH n, labels(n) AS lbls\n"
    "WITH n, lbls,\n"
    "     CASE\n"
    "         WHEN 'Law' IN lbls THEN coalesce(n.law_name, n.name, '')\n"
    "         WHEN 'Article' IN lbls THEN coalesce(n.title, n.content, '')\n"
    "         WHEN 'Clause' IN lbls THEN coalesce(n.content, '')\n"
    "         WHEN 'Point' IN lbls THEN coalesce(n.content, '')\n"
    "         WHEN 'Document' IN lbls THEN coalesce(n.name, '')\n"
    "         ELSE coalesce(n.name, n.title, n.content, '')\n"
    "     END AS primary_text,\n"
    "     CASE\n"
    "         WHEN 'Law' IN lbls THEN coalesce(n.law_id, n.law_name, '')\n"
    "         WHEN 'Article' IN lbls THEN coalesce(n.article_key, '')\n"
    "         WHEN 'Clause' IN lbls THEN coalesce(n.clause_key, '')\n"
    "         WHEN 'Point' IN lbls THEN coalesce(n.point_key, '')\n"
    "         WHEN 'Document' IN lbls THEN coalesce(n.name, '')\n"
    "         ELSE ''\n"
    "     END AS identifier\n"
    "WITH id(n) AS node_id, lbls AS labels, primary_text, identifier,\n"
    "     CASE\n"
    "         WHEN $term = '' THEN 1\n"
    "         WHEN toLower(primary_text) CONTAINS $term OR toLower(identifier) CONTAINS $term THEN 1\n"
    "         ELSE 0\n"
    "     END AS score\n"
    "WHERE score > 0 OR $term = ''\n"
    "RETURN node_id, labels, identifier, primary_text AS text, score\n"
    "ORDER BY score DESC, text ASC\n"
    "LIMIT $limit"
)

NODE_CONTEXT = (
    "MATCH (n) WHERE id(n) = $node_id\n"
    "MATCH (n)-[r]->(m)\n"
    "RETURN 'out' AS direction, type(r) AS rel_type, id(m) AS other_id, labels(m) AS labels, "
    "       CASE\n"
    "           WHEN 'Law' IN labels(m) THEN coalesce(m.law_name, m.name, '')\n"
    "           WHEN 'Article' IN labels(m) THEN coalesce(m.title, m.content, '')\n"
    "           WHEN 'Clause' IN labels(m) THEN coalesce(m.content, '')\n"
    "           WHEN 'Point' IN labels(m) THEN coalesce(m.content, '')\n"
    "           WHEN 'Document' IN labels(m) THEN coalesce(m.name, '')\n"
    "           ELSE coalesce(m.name, m.title, m.content, '')\n"
    "       END AS text,\n"
    "       CASE\n"
    "           WHEN 'Law' IN labels(m) THEN coalesce(m.law_id, m.law_name, '')\n"
    "           WHEN 'Article' IN labels(m) THEN coalesce(m.article_key, '')\n"
    "           WHEN 'Clause' IN labels(m) THEN coalesce(m.clause_key, '')\n"
    "           WHEN 'Point' IN labels(m) THEN coalesce(m.point_key, '')\n"
    "           WHEN 'Document' IN labels(m) THEN coalesce(m.name, '')\n"
    "           ELSE ''\n"
    "       END AS identifier, r.effective_date AS effective_date, r.expiry_date AS expiry_date, r.description AS description\n"
    "UNION ALL\n"
    "MATCH (m)-[r]->(n) WHERE id(n) = $node_id\n"
    "RETURN 'in' AS direction, type(r) AS rel_type, id(m) AS other_id, labels(m) AS labels, "
    "       CASE\n"
    "           WHEN 'Law' IN labels(m) THEN coalesce(m.law_name, m.name, '')\n"
    "           WHEN 'Article' IN labels(m) THEN coalesce(m.title, m.content, '')\n"
    "           WHEN 'Clause' IN labels(m) THEN coalesce(m.content, '')\n"
    "           WHEN 'Point' IN labels(m) THEN coalesce(m.content, '')\n"
    "           WHEN 'Document' IN labels(m) THEN coalesce(m.name, '')\n"
    "           ELSE coalesce(m.name, m.title, m.content, '')\n"
    "       END AS text,\n"
    "       CASE\n"
    "           WHEN 'Law' IN labels(m) THEN coalesce(m.law_id, m.law_name, '')\n"
    "           WHEN 'Article' IN labels(m) THEN coalesce(m.article_key, '')\n"
    "           WHEN 'Clause' IN labels(m) THEN coalesce(m.clause_key, '')\n"
    "           WHEN 'Point' IN labels(m) THEN coalesce(m.point_key, '')\n"
    "           WHEN 'Document' IN labels(m) THEN coalesce(m.name, '')\n"
    "           ELSE ''\n"
    "       END AS identifier, r.effective_date AS effective_date, r.expiry_date AS expiry_date, r.description AS description\n"
    "LIMIT $relation_limit"
)

STRUCTURED_LAW_QUERY = (
    "MATCH (l:Law)\n"
    "WHERE (size($law_ids) > 0 AND toUpper(coalesce(l.law_id,'')) IN $law_ids) "
    "   OR (size($law_names) > 0 AND toLower(coalesce(l.law_name,'')) IN $law_names)\n"
    "RETURN id(l) AS node_id, labels(l) AS labels, coalesce(l.law_id, l.law_name, '') AS identifier, coalesce(l.law_name, '') AS text"
)

STRUCTURED_ARTICLE_QUERY = (
    "MATCH (l:Law)-[:HAS_ARTICLE]->(a:Article)\n"
    "WHERE ((size($law_ids) > 0 AND toUpper(coalesce(l.law_id,'')) IN $law_ids) "
    "    OR (size($law_names) > 0 AND toLower(coalesce(l.law_name,'')) IN $law_names))\n"
    "  AND (size($article_numbers) = 0 OR a.article_number IN $article_numbers)\n"
    "RETURN id(a) AS node_id, labels(a) AS labels, coalesce(a.article_key, '') AS identifier, coalesce(a.title, a.content, '') AS text, a.article_number AS article_number, toUpper(l.law_id) AS law_id"
)

STRUCTURED_CLAUSE_QUERY = (
    "MATCH (l:Law)-[:HAS_ARTICLE]->(a:Article)-[:HAS_CLAUSE]->(c:Clause)\n"
    "WHERE ((size($law_ids) > 0 AND toUpper(coalesce(l.law_id,'')) IN $law_ids) "
    "    OR (size($law_names) > 0 AND toLower(coalesce(l.law_name,'')) IN $law_names))\n"
    "  AND (size($article_numbers) = 0 OR a.article_number IN $article_numbers)\n"
    "  AND (size($clause_numbers) = 0 OR c.clause_number IN $clause_numbers)\n"
    "RETURN id(c) AS node_id, labels(c) AS labels, coalesce(c.clause_key, '') AS identifier, coalesce(c.content, '') AS text, a.article_number AS article_number, c.clause_number AS clause_number, toUpper(l.law_id) AS law_id"
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

