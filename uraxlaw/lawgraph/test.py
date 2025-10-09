"""
GraphRAG Pháp lý Việt Nam — Neo4j + ChatGPT + PydanticAI (end‑to‑end demo)

What you get:
1) Pydantic models (LawSchema/Article/Clause/Relation) for strict validation
2) PydanticAI Agent that calls ChatGPT to EXTRACT → validated JSON
3) Neo4j ingestion (MERGE nodes/edges) with temporal attributes (effective/expiry)
4) Tiny demo: ingest sample text (Luật GTGT) and query "Điều 1 làm gì?"

Prerequisites:
  pip install pydantic pydantic-ai neo4j python-dateutil
  # If you use OpenAI's official SDK for PydanticAI provider:
  pip install openai

ENV needed:
  OPENAI_API_KEY=...
  NEO4J_URI=bolt://localhost:7687
  NEO4J_USER=neo4j
  NEO4J_PASSWORD=your_password

Run:
  python neo4j_graphrag_vn_law.py
"""
from __future__ import annotations
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
import argparse
import os
import json
import re
import unicodedata
from datetime import datetime

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from neo4j import GraphDatabase
from dotenv import load_dotenv

from .models import Relation, Clause, Point, Article, LawSchema, QuerySlots

load_dotenv()

LAW_ID_PATTERN = re.compile(r"\d{1,3}/\d{4}/[A-Z0-9ĐÂĂÊÔƠƯ-]+", re.IGNORECASE)

REL_KEYWORDS = {
    "sửa đổi": "MODIFIES",
    "sua doi": "MODIFIES",
    "bổ sung": "ADDS",
    "bo sung": "ADDS",
    "bãi bỏ": "REPEALS",
    "bai bo": "REPEALS",
    "thay thế": "REPLACES",
    "thay the": "REPLACES",
    "dẫn chiếu": "REFERS_TO",
    "dan chieu": "REFERS_TO",
    "định nghĩa": "DEFINES",
    "dinh nghia": "DEFINES",
}

ALIAS_GROUPS = [
    {
        "law_id": "13/2008/QH12",
        "law_name": "Luật thuế giá trị gia tăng",
        "aliases": [
            "luật thuế giá trị gia tăng",
            "luat thue gia tri gia tang",
            "luật thuế gtgt",
            "luat thue gtgt",
            "thuế gtgt",
        ],
    },
    {
        "law_id": "15/2009/QH12",
        "law_name": "Luật sửa đổi, bổ sung một số điều của Luật thuế GTGT",
        "aliases": [
            "luật sửa đổi bổ sung luật thuế gtgt",
            "luat sua doi bo sung luat thue gtgt",
            "luật sửa đổi, bổ sung một số điều của luật thuế gtgt",
            "luat sua doi 15/2009/qh12",
        ],
    },
]


def _strip_accents(text: str) -> str:
    nfkd = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in nfkd if unicodedata.category(ch) != "Mn")


ALIAS_LOOKUP: Dict[str, Tuple[str, str]] = {}
for group in ALIAS_GROUPS:
    law_id = group["law_id"]
    law_name = group["law_name"]
    for alias in group["aliases"]:
        normalized_alias = _strip_accents(alias).lower()
        ALIAS_LOOKUP[normalized_alias] = (law_id, law_name)


def normalize_text(text: str) -> str:
    return _strip_accents(text).lower()


def extract_law_ids(*texts: Optional[str]) -> List[str]:
    seen: List[str] = []
    for text in texts:
        if not text:
            continue
        for match in LAW_ID_PATTERN.findall(text):
            normalized = match.upper()
            if normalized not in seen:
                seen.append(normalized)
    return seen


def extract_query_slots(question: str) -> QuerySlots:
    normalized = normalize_text(question)
    slots = QuerySlots()

    slots.law_ids.extend(extract_law_ids(question))

    for alias, (law_id, law_name) in ALIAS_LOOKUP.items():
        if alias in normalized:
            if law_id not in slots.law_ids:
                slots.law_ids.append(law_id)
            if law_name not in slots.law_names:
                slots.law_names.append(law_name)

    article_matches = re.findall(r"(?i)(?:điều|article)\s*(\d+)", question)
    for match in article_matches:
        try:
            num = int(match)
            if num not in slots.article_numbers:
                slots.article_numbers.append(num)
        except ValueError:
            pass

    clause_matches = re.findall(r"(?i)(?:khoản|clause)\s*(\d+)", question)
    for match in clause_matches:
        try:
            num = int(match)
            if num not in slots.clause_numbers:
                slots.clause_numbers.append(num)
        except ValueError:
            pass

    date_match = re.search(r"(\d{1,2}/\d{1,2}/\d{4})", question)
    if date_match:
        try:
            slots.as_of = datetime.strptime(date_match.group(1), "%d/%m/%Y").date().isoformat()
        except ValueError:
            slots.as_of = None
    else:
        iso_match = re.search(r"(\d{4}-\d{2}-\d{2})", question)
        if iso_match:
            slots.as_of = iso_match.group(1)

    rel_types_found: List[str] = []
    for key, rel_type in REL_KEYWORDS.items():
        if key in normalized and rel_type not in rel_types_found:
            rel_types_found.append(rel_type)
    slots.rel_types = rel_types_found

    cleaned = normalized
    for alias in ALIAS_LOOKUP.keys():
        cleaned = cleaned.replace(alias, " ")
    for match in slots.law_ids:
        cleaned = cleaned.replace(match.lower(), " ")
    cleaned = re.sub(r"(?i)(điều|article)\s*\d+", " ", cleaned)
    cleaned = re.sub(r"(?i)(khoản|clause)\s*\d+", " ", cleaned)
    cleaned = re.sub(r"(?i)(điểm|point)\s*[a-z0-9]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if cleaned:
        slots.open_text = cleaned
        slots.keywords = [word for word in cleaned.split(" ") if word]

    slots.is_ambiguous = not slots.law_ids and not slots.law_names

    return slots
# =======================e
# 0) Pydantic models: Schema with date normalization
# (moved to uraxlaw.lawgraph.models)
# ========================
# (Removed in-file model definitions; imported above.)


@dataclass
class Neo4jConfig:
    uri: str
    user: str
    password: str


def get_driver(cfg: Neo4jConfig):
    return GraphDatabase.driver(cfg.uri, auth=(cfg.user, cfg.password))

SCHEMA_CYPHER = [
    # Constraints & indexes
    "CREATE CONSTRAINT law_id_unique IF NOT EXISTS FOR (l:Law) REQUIRE l.law_id IS UNIQUE",
    "CREATE INDEX law_name_idx IF NOT EXISTS FOR (l:Law) ON (l.law_name)",
    "CREATE INDEX article_key_idx IF NOT EXISTS FOR (a:Article) ON (a.article_key)",
    "CREATE INDEX clause_key_idx IF NOT EXISTS FOR (c:Clause) ON (c.clause_key)",
    "CREATE INDEX point_key_idx IF NOT EXISTS FOR (p:Point) ON (p.point_key)",
]

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
    )
}


def init_schema(driver):
    with driver.session() as s:
        for cy in SCHEMA_CYPHER:
            s.run(cy)


def ingest_to_neo4j(driver, payload: LawSchema):
    law_id = payload.law_id or payload.law_name

    def normalize_point_symbol(symbol: Optional[str]) -> str:
        if symbol is None:
            return ""
        sym = symbol.strip()
        sym = sym.rstrip(").")
        sym = sym.replace(" ", "")
        return sym.upper()

    with driver.session() as s:
        def ingest_relation_entry(
            rel_obj: Relation,
            fallback_source: Optional[str],
            source_node_key: Optional[str],
            source_node_type: Optional[str],
        ):
            if not rel_obj or not rel_obj.type:
                return

            tgt_name_raw = rel_obj.target or ""
            tgt_name = tgt_name_raw.strip() or None
            target_point_key = None
            target_clause_key = None
            target_article_key = None

            referenced_law_ids = extract_law_ids(tgt_name_raw, rel_obj.source, rel_obj.description)
            base_key_prefixes: List[Optional[str]] = []
            base_key_prefixes.extend(referenced_law_ids)
            if law_id and law_id not in base_key_prefixes:
                base_key_prefixes.append(law_id)
            if payload.law_id and payload.law_id not in base_key_prefixes:
                base_key_prefixes.append(payload.law_id)
            if payload.law_name and payload.law_name not in base_key_prefixes:
                base_key_prefixes.append(payload.law_name)

            def _match_direct_key(candidate: str) -> bool:
                nonlocal target_point_key, target_clause_key, target_article_key
                if not candidate:
                    return False
                for base in base_key_prefixes:
                    if not base:
                        continue
                    pattern = rf"^\s*{re.escape(base)}::(?:Điều|Article)#(\d+)(?:::(?:Khoản|Clause)#(\d+))?(?:::(?:Điểm|Point)#([A-Za-z0-9]+))?\s*$"
                    m = re.match(pattern, candidate)
                    if not m:
                        continue
                    article_no = int(m.group(1))
                    target_article_key = f"{base}::Điều#{article_no}"
                    clause_group = m.group(2)
                    point_group = m.group(3)
                    if clause_group is not None:
                        clause_no = int(clause_group)
                        target_clause_key = f"{target_article_key}::Khoản#{clause_no}"
                    if point_group is not None and target_clause_key:
                        point_symbol_norm = normalize_point_symbol(point_group)
                        if point_symbol_norm:
                            target_point_key = f"{target_clause_key}::Điểm#{point_symbol_norm}"
                    return True
                return False

            if tgt_name and _match_direct_key(tgt_name_raw):
                pass

            article_match = re.search(r"(?:[Đđ]iều|Article)\s*(\d+)", tgt_name_raw)
            clause_match = re.search(r"(?:[Kk]hoản|Clause)\s*(\d+)", tgt_name_raw)
            point_match = re.search(r"(?:[Đđ]iểm|Point)\s*([A-Za-z0-9]+)", tgt_name_raw)

            try:
                if point_match and clause_match and article_match:
                    clause_no = int(clause_match.group(1))
                    article_no = int(article_match.group(1))
                    point_symbol_norm = normalize_point_symbol(point_match.group(1))
                    for base in base_key_prefixes:
                        if not base or not point_symbol_norm:
                            continue
                        target_article_key = f"{base}::Điều#{article_no}"
                        target_clause_key = f"{target_article_key}::Khoản#{clause_no}"
                        target_point_key = f"{target_clause_key}::Điểm#{point_symbol_norm}"
                        break
                elif clause_match and article_match:
                    clause_no = int(clause_match.group(1))
                    article_no = int(article_match.group(1))
                    for base in base_key_prefixes:
                        if not base:
                            continue
                        target_article_key = f"{base}::Điều#{article_no}"
                        target_clause_key = f"{target_article_key}::Khoản#{clause_no}"
                        break
                elif article_match:
                    article_no = int(article_match.group(1))
                    for base in base_key_prefixes:
                        if not base:
                            continue
                        target_article_key = f"{base}::Điều#{article_no}"
                        break
            except ValueError:
                target_point_key = target_clause_key = target_article_key = None

            if not any([target_point_key, target_clause_key, target_article_key]) and tgt_name is None:
                return

            raw_source_name = rel_obj.source or fallback_source or payload.law_name or payload.law_id or None
            source_name = raw_source_name.strip() if isinstance(raw_source_name, str) and raw_source_name.strip() else None

            source_point_key = source_node_key if source_node_type == "POINT" else None
            source_clause_key = source_node_key if source_node_type == "CLAUSE" else None
            source_article_key = source_node_key if source_node_type == "ARTICLE" else None

            if source_name is None and not any([source_point_key, source_clause_key, source_article_key]):
                return

            s.run(INGEST_CYPHER["relation"], {
                "type": rel_obj.type,
                "source_name": source_name,
                "source_point_key": source_point_key,
                "source_clause_key": source_clause_key,
                "source_article_key": source_article_key,
                "target_point_key": target_point_key,
                "target_clause_key": target_clause_key,
                "target_article_key": target_article_key,
                "target_name": tgt_name,
                "effective_date": rel_obj.effective_date,
                "expiry_date": rel_obj.expiry_date,
                "description": rel_obj.description,
            })

        # Upsert Law
        s.run(INGEST_CYPHER["law"], {
            "law_id": payload.law_id,
            "law_name": payload.law_name,
            "document_type": payload.document_type,
            "issued_by": payload.issued_by,
            "signer": payload.signer,
            "issued_date": payload.issued_date,
            "promulgation_date": payload.promulgation_date,
            "effective_date": payload.effective_date,
            "expiry_date": payload.expiry_date,
            "scope": payload.scope,
            "language": payload.language,
            "aliases": payload.aliases,
            "modified_by": payload.modified_by,
        })

        default_source = payload.law_name or payload.law_id

        # Articles
        for art in payload.articles:
            article_key = f"{law_id}::Điều#{art.article_number}"
            s.run(INGEST_CYPHER["article"], {
                "law_id": payload.law_id,
                "law_name": payload.law_name,
                "article_key": article_key,
                "article_number": art.article_number,
                "title": art.title,
                "content": art.content,
                "effective_date": art.effective_date,
                "expiry_date": art.expiry_date,
            })

            # Clauses
            for cl in art.clauses:
                clause_key = f"{article_key}::Khoản#{cl.clause_number}"
                s.run(INGEST_CYPHER["clause"], {
                    "article_key": article_key,
                    "clause_key": clause_key,
                    "clause_number": cl.clause_number,
                    "content": cl.content,
                    "effective_date": cl.effective_date,
                    "expiry_date": cl.expiry_date,
                })

                # Points (Điểm)
                for pt in cl.points:
                    point_symbol_norm = normalize_point_symbol(pt.point_symbol)
                    if not point_symbol_norm:
                        continue
                    point_key = f"{clause_key}::Điểm#{point_symbol_norm}"
                    s.run(INGEST_CYPHER["point"], {
                        "clause_key": clause_key,
                        "point_key": point_key,
                        "point_symbol": pt.point_symbol or point_symbol_norm,
                        "point_symbol_norm": point_symbol_norm,
                        "content": pt.content,
                        "effective_date": pt.effective_date,
                        "expiry_date": pt.expiry_date,
                    })

                    for rel in pt.relations:
                        ingest_relation_entry(rel, default_source, point_key, "POINT")

                # Clause-level relations
                for rel in cl.relations:
                    ingest_relation_entry(rel, default_source, clause_key, "CLAUSE")

# ========================
# 3) Query helpers
# ========================
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


def ask_what_article_does(driver, law_name: str, article_number: int) -> str:
    with driver.session() as s:
        rec = s.run(GET_ARTICLE_CONTENT, {"law_name": law_name, "article_number": article_number}).single()
        if not rec:
            return "Không tìm thấy điều luật."
        title = rec["title"]
        content = rec["content"]
        eff = rec["law_effective"]
        out = [f"Điều {article_number} của {law_name}"]
        if title:
            out.append(f"(Tiêu đề: {title})")
        if content:
            out.append(f"nêu: {content}")
        if eff:
            out.append(f"(Luật có hiệu lực từ {eff}).")
        # modifiers (if any)
        mods = s.run(GET_MODIFIERS, {"law_name": law_name, "article_number": article_number}).data()
        mods = [m for m in mods if m["source"]]
        if mods:
            out.append("\nCác văn bản sửa đổi điều này:")
            for m in mods:
                eff_m = m.get("effective")
                out.append(f"- {m['source']} (hiệu lực từ {eff_m})")
        return " ".join(out)


def search_articles(driver, question: str, limit: int = 5) -> List[Dict[str, Any]]:
    term = re.sub(r"\s+", " ", question).strip().lower()
    article_numbers = [int(n) for n in re.findall(r"(?i)(?:điều|article)\s*(\d+)", question)]
    law_ids = extract_law_ids(question)
    law_name_term = ""
    law_name_match = re.search(r"(?i)luật\s+([^\d,;:\n\?]+)", question)
    if law_name_match:
        law_name_term = re.sub(r"\s+", " ", law_name_match.group(1)).strip().lower()

    cleaned_term = term
    cleaned_term = re.sub(r"(?i)(điều|article)\s*\d+", " ", cleaned_term)
    cleaned_term = re.sub(r"(?i)(khoản|clause)\s*\d+", " ", cleaned_term)
    cleaned_term = re.sub(r"(?i)(điểm|point)\s*[a-z0-9]+", " ", cleaned_term)
    for lid in law_ids:
        cleaned_term = cleaned_term.replace(lid.lower(), " ")
    if law_name_term:
        cleaned_term = cleaned_term.replace(law_name_term, " ")
    cleaned_term = re.sub(r"\s+", " ", cleaned_term).strip()

    params = {
        "article_numbers": article_numbers,
        "law_ids": [lid.upper() for lid in law_ids],
        "law_name_term": law_name_term,
        "term": cleaned_term,
        "limit": max(1, limit),
    }

    with driver.session() as s:
        records = s.run(SEARCH_TOP_ARTICLES, params).data()

    return records


def search_all_nodes(driver, question: str, limit: int = 10) -> List[Dict[str, Any]]:
    term = re.sub(r"\s+", " ", question).strip().lower()
    params = {
        "term": term,
        "limit": max(1, limit),
    }
    with driver.session() as s:
        recs = s.run(SEARCH_ALL_NODES, params).data()
    return recs


def structured_candidates(driver, slots: QuerySlots) -> Dict[int, Dict[str, Any]]:
    params_common = {
        "law_ids": [law_id.upper() for law_id in slots.law_ids],
        "law_names": [name.lower() for name in slots.law_names],
        "article_numbers": slots.article_numbers,
        "clause_numbers": slots.clause_numbers,
    }

    candidates: Dict[int, Dict[str, Any]] = {}

    def _register(record: Dict[str, Any], extra_score: float, reason: str) -> None:
        node_id = record["node_id"]
        entry = candidates.setdefault(
            node_id,
            {
                "node_id": node_id,
                "labels": record.get("labels", []),
                "identifier": record.get("identifier"),
                "text": record.get("text"),
                "score": 0.0,
                "reasons": [],
            },
        )
        entry["score"] += extra_score
        entry["reasons"].append(reason)

    with driver.session() as session:
        if params_common["law_ids"] or params_common["law_names"]:
            law_records = session.run(STRUCTURED_LAW_QUERY, params_common).data()
            for rec in law_records:
                _register(rec, 8.0, "trùng luật")

        if params_common["law_ids"] or params_common["law_names"]:
            article_records = session.run(STRUCTURED_ARTICLE_QUERY, params_common).data()
            for rec in article_records:
                bonus = 6.0
                if params_common["article_numbers"]:
                    bonus += 4.0
                _register(rec, bonus, "ứng viên điều" + (f" {rec.get('article_number')}" if rec.get("article_number") else ""))

        if params_common["clause_numbers"] or params_common["article_numbers"]:
            clause_records = session.run(STRUCTURED_CLAUSE_QUERY, params_common).data()
            for rec in clause_records:
                bonus = 5.0
                if params_common["clause_numbers"]:
                    bonus += 3.0
                if params_common["article_numbers"]:
                    bonus += 2.0
                clause_desc = ""
                if rec.get("article_number") is not None and rec.get("clause_number") is not None:
                    clause_desc = f" điều {rec['article_number']} khoản {rec['clause_number']}"
                _register(rec, bonus, f"ứng viên khoản{clause_desc}")

    return candidates


def fetch_node_context(driver, node_id: int, relation_limit: int = 10) -> List[Dict[str, Any]]:
    with driver.session() as session:
        return session.run(
            NODE_CONTEXT,
            {"node_id": node_id, "relation_limit": relation_limit},
        ).data()


def route_query(driver, question: str, limit: int = 5) -> Dict[str, Any]:
    slots = extract_query_slots(question)
    combined: Dict[int, Dict[str, Any]] = structured_candidates(driver, slots)

    all_nodes = search_all_nodes(driver, slots.open_text or question, limit=50)
    for rec in all_nodes:
        node_id = rec.get("node_id")
        if node_id is None:
            continue
        entry = combined.setdefault(
            node_id,
            {
                "node_id": node_id,
                "labels": rec.get("labels", []),
                "identifier": rec.get("identifier"),
                "text": rec.get("text"),
                "score": 0.0,
                "reasons": [],
            },
        )
        entry["score"] += float(rec.get("score", 0.0))
        entry["reasons"].append("khớp toàn văn")

    ranked = sorted(
        combined.values(),
        key=lambda item: item["score"],
        reverse=True,
    )

    for item in ranked[:limit]:
        node_id = item["node_id"]
        item["context"] = fetch_node_context(driver, node_id)

    return {
        "slots": slots,
        "results": ranked[:limit],
    }


def _snippet(text: Optional[str], limit: int = 180) -> str:
    if not text:
        return "(không có nội dung)"
    clean = re.sub(r"\s+", " ", text).strip()
    if len(clean) <= limit:
        return clean
    return clean[:limit].rstrip() + "…"

# ========================
# 4) Sample text & demo runner
# ========================


SAMPLE_TEXT_1 = """
QUỐC HỘI
_________________
Luật số: 13/2008/QH12    CỘNG HOÀ XÃ HỘI CHỦ NGHĨA VIỆT NAM
Độc lập - Tự do - Hạnh phúc
___________________________

LUẬT THUẾ GIÁ TRỊ GIA TĂNG

Điều 2. Thuế giá trị gia tăng
1. Thuế giá trị gia tăng là thuế tính trên giá trị tăng thêm của hàng hoá, dịch vụ.
2. Đối tượng nộp thuế là tổ chức, cá nhân sản xuất, kinh doanh hàng hoá, dịch vụ chịu thuế.
3. Quy định chuyển tiếp áp dụng đến hết ngày 31/12/2009.
Ghi chú: Khoản 1 Điều 2 định nghĩa "thuế giá trị gia tăng". Văn bản này có hiệu lực từ 01/01/2009.
"""

SAMPLE_TEXT_2 = """
QUỐC HỘI
_________________
Luật số: 15/2009/QH12    CỘNG HOÀ XÃ HỘI CHỦ NGHĨA VIỆT NAM
Độc lập - Tự do - Hạnh phúc
___________________________

LUẬT SỬA ĐỔI, BỔ SUNG MỘT SỐ ĐIỀU CỦA LUẬT THUẾ GTGT

Điều 1. Sửa đổi, bổ sung Luật số 13/2008/QH12
1. Sửa đổi Khoản 1 Điều 2 của Luật số 13/2008/QH12 như sau:
   "Thuế giá trị gia tăng là thuế tính trên giá trị tăng thêm của hàng hoá, dịch vụ phát sinh trong quá trình sản xuất, lưu thông và tiêu dùng."
2. Bổ sung Khoản 5 vào Điều 2 của Luật số 13/2008/QH12:
   "Khoản 5. Giá trị tăng thêm là phần chênh lệch giữa giá bán và giá mua hợp pháp."
3. Bãi bỏ Khoản 3 Điều 2 của Luật số 13/2008/QH12.
4. Điều khoản thi hành:
   a) Luật này có hiệu lực từ ngày 01/01/2010.
   b) Các quy định trước đây trái với Luật này hết hiệu lực kể từ ngày Luật này có hiệu lực.

"""

# Chạy demo với 2 sample
SAMPLE_TEXTS = [SAMPLE_TEXT_1, SAMPLE_TEXT_2]


import asyncio

def ensure_env() -> Neo4jConfig:
    cfg = Neo4jConfig(
        uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        user=os.getenv("NEO4J_USER", "neo4j"),
        password=os.getenv("NEO4J_PASSWORD", "password"),
    )
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set")
    return cfg

async def demo(query: Optional[str] = None, limit: int = 5, interactive: bool = False):
    cfg = ensure_env()
    driver = get_driver(cfg)
    init_schema(driver)

    # 1) Extract schema via PydanticAI (ChatGPT)
    print("\n=== Extracting JSON schema via PydanticAI + ChatGPT ===")
    law_schema: LawSchema = await extract_schema(SAMPLE_TEXT_2)
    print(json.dumps(json.loads(law_schema.model_dump_json()), ensure_ascii=False, indent=2))

    # 2) Ingest to Neo4j
    print("\n=== Ingesting into Neo4j ===")

    ingest_to_neo4j(driver, law_schema)

    # 3) Search all nodes demo
    search_terms: List[str] = []
    if query:
        search_terms.append(query)

    if interactive:
        try:
            while True:
                user_q = input("\nNhập từ khóa (Enter để thoát): ").strip()
                if not user_q:
                    break
                search_terms.append(user_q)
        except (EOFError, KeyboardInterrupt):
            print("\n(thoát chế độ tương tác)")

    if not search_terms:
        search_terms.append("LUẬT THUẾ GIÁ TRỊ GIA TĂNG Điều 2 Khoản 3 ")

    for q in search_terms:
        print(f"\n=== Tìm kiếm top {limit} nút cho: {q} ===")
        routed = route_query(driver, q, limit=limit)
        results = routed["results"]
        slots = routed["slots"]
        if slots.is_ambiguous:
            print(" Thiếu thông tin luật cụ thể, kết quả có thể rộng.")
        if not results:
            print("(không tìm thấy kết quả)")
            continue
        for idx, hit in enumerate(results, start=1):
            labels = ",".join(hit.get("labels", []))
            identifier = hit.get("identifier") or ""
            text = hit.get("text") or ""
            snippet = _snippet(text)
            id_part = f" [{identifier}]" if identifier else ""
            reasons = "; ".join(hit.get("reasons", []))
            print(f"{idx}. ({labels}){id_part}: {snippet} (score={hit['score']:.1f})")
            if reasons:
                print(f"   Lý do: {reasons}")
            context = hit.get("context") or []
            if context:
                print("   Quan hệ liên quan:")
                for rel in context[:limit]:
                    direction = "→" if rel.get("direction") == "out" else "←"
                    rel_type = rel.get("rel_type") or "?"
                    other_id = rel.get("other_id")
                    other_labels = ",".join(rel.get("labels", []))
                    other_identifier = rel.get("identifier") or ""
                    other_text = _snippet(rel.get("text"))
                    eff = rel.get("effective_date")
                    exp = rel.get("expiry_date")
                    desc = rel.get("description")
                    meta_parts = []
                    if eff:
                        meta_parts.append(f"hiệu lực {eff}")
                    if exp:
                        meta_parts.append(f"hết hiệu lực {exp}")
                    if desc:
                        meta_parts.append(desc)
                    meta = f" ({'; '.join(meta_parts)})" if meta_parts else ""
                    other_id_part = f" [{other_identifier}]" if other_identifier else ""
                    print(
                        f"      {direction} {rel_type}: ({other_labels}){other_id_part} {other_text}{meta}"
                    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo tìm kiếm nút trong đồ thị pháp luật")
    parser.add_argument("query", nargs="?", help="Từ khóa tìm kiếm ban đầu")
    parser.add_argument("--limit", type=int, default=5, help="Số kết quả tối đa")
    parser.add_argument("--interactive", action="store_true", help="Bật chế độ nhập nhiều truy vấn")
    args = parser.parse_args()

    asyncio.run(demo(query=args.query, limit=args.limit, interactive=args.interactive))
