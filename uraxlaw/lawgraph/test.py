"""
GraphRAG Pháp lý Việt Nam — Neo4j + ChatGPT + PydanticAI (end‑to‑end demo)

What you get:
1) Pydantic models (LawSchema/Article/Clause/Relation) for strict validation
2) PydanticAI Agent that calls ChatGPT to EXTRACT → validated JSON
3) Neo4j ingestion (MERGE nodes/edges) with temporal attributes (effective/expiry)
4) Tiny demo: ingest sample text (Luật GTGT) and query "Điều 1 làm gì?"
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
import asyncio

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from neo4j import GraphDatabase
from dotenv import load_dotenv

from .models import Relation, Clause, Point, Article, LawSchema, QuerySlots
from .neo4j_db import (
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

load_dotenv()

LAW_ID_PATTERN = re.compile(r"\d{1,3}/\d{4}/[A-Z0-9ĐÂĂÊÔƠƯ-]+", re.IGNORECASE)
REL_KEYWORDS = {
    "sửa đổi": "MODIFIES","sua doi": "MODIFIES","bổ sung": "ADDS","bo dung": "ADDS","bo sung": "ADDS", "bãi bỏ": "REPEALS","bai bo": "REPEALS","thay thế": "REPLACES","thay the": "REPLACES","dẫn chiếu": "REFERS_TO","dan chieu": "REFERS_TO","định nghĩa": "DEFINES","dinh nghia": "DEFINES",
}
ALIAS_GROUPS = [
    {"law_id": "13/2008/QH12","law_name": "Luật thuế giá trị gia tăng","aliases": ["luật thuế giá trị gia tăng","luat thue gia tri gia tang","luật thuế gtgt","luat thue gtgt","thuế gtgt"]},
    {"law_id": "15/2009/QH12","law_name": "Luật sửa đổi, bổ sung một số điều của Luật thuế GTGT","aliases": ["luật sửa đổi bổ sung luật thuế gtgt","luat sua doi bo sung luat thue gtgt","luật sửa đổi, bổ sung một số điều của luật thuế gtgt","luat sua doi 15/2009/qh12"]},
]

def _strip_accents(text: str) -> str:
    nfkd = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in nfkd if unicodedata.category(ch) != "Mn")

ALIAS_LOOKUP: Dict[str, Tuple[str, str]] = {}
for group in ALIAS_GROUPS:
    for alias in group["aliases"]:
        ALIAS_LOOKUP[_strip_accents(alias).lower()] = (group["law_id"], group["law_name"])

def normalize_text(text: str) -> str: return _strip_accents(text).lower()

def extract_law_ids(*texts: Optional[str]) -> List[str]:
    seen: List[str] = []
    for text in texts:
        if not text: continue
        for match in LAW_ID_PATTERN.findall(text):
            m_up = match.upper()
            if m_up not in seen: seen.append(m_up)
    return seen

def extract_query_slots(question: str) -> QuerySlots:
    normalized = normalize_text(question)
    slots = QuerySlots()
    slots.law_ids.extend(extract_law_ids(question))
    for alias, (law_id, law_name) in ALIAS_LOOKUP.items():
        if alias in normalized:
            if law_id not in slots.law_ids: slots.law_ids.append(law_id)
            if law_name not in slots.law_names: slots.law_names.append(law_name)
    for match in re.findall(r"(?i)(?:điều|article)\s*(\d+)", question):
        try:
            num = int(match)
            if num not in slots.article_numbers: slots.article_numbers.append(num)
        except ValueError: pass
    for match in re.findall(r"(?i)(?:khoản|clause)\s*(\d+)", question):
        try:
            num = int(match)
            if num not in slots.clause_numbers: slots.clause_numbers.append(num)
        except ValueError: pass
    date_match = re.search(r"(\d{1,2}/\d{1,2}/\d{4})", question)
    if date_match:
        try: slots.as_of = datetime.strptime(date_match.group(1), "%d/%m/%Y").date().isoformat()
        except ValueError: slots.as_of = None
    else:
        iso_match = re.search(r"(\d{4}-\d{2}-\d{2})", question)
        if iso_match: slots.as_of = iso_match.group(1)
    rel_types_found: List[str] = []
    for key, rel_type in REL_KEYWORDS.items():
        if key in normalized and rel_type not in rel_types_found: rel_types_found.append(rel_type)
    slots.rel_types = rel_types_found
    cleaned = normalized
    for alias in ALIAS_LOOKUP.keys(): cleaned = cleaned.replace(alias, " ")
    for match in slots.law_ids: cleaned = cleaned.replace(match.lower(), " ")
    cleaned = re.sub(r"(?i)(điều|article)\s*\d+", " ", cleaned)
    cleaned = re.sub(r"(?i)(khoản|clause)\s*\d+", " ", cleaned)
    cleaned = re.sub(r"(?i)(điểm|point)\s*[a-z0-9]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if cleaned:
        slots.open_text = cleaned
        slots.keywords = [w for w in cleaned.split(" ") if w]
    slots.is_ambiguous = not slots.law_ids and not slots.law_names
    return slots

@dataclass
class Neo4jConfig: uri: str; user: str; password: str

def get_driver(cfg: Neo4jConfig): return GraphDatabase.driver(cfg.uri, auth=(cfg.user, cfg.password))

def init_schema(driver):
    with driver.session() as s:
        for cy in SCHEMA_CYPHER: s.run(cy)

def ingest_to_neo4j(driver, payload: LawSchema):
    law_id = payload.law_id or payload.law_name
    def normalize_point_symbol(symbol: Optional[str]) -> str:
        if symbol is None: return ""
        sym = symbol.strip().rstrip(").").replace(" ", "").upper()
        return sym
    with driver.session() as s:
        def ingest_relation_entry(rel_obj: Relation, fallback_source: Optional[str], source_node_key: Optional[str], source_node_type: Optional[str]):
            if not rel_obj or not rel_obj.type: return
            tgt_name_raw = rel_obj.target or ""; tgt_name = tgt_name_raw.strip() or None
            target_point_key = target_clause_key = target_article_key = None
            referenced_law_ids = extract_law_ids(tgt_name_raw, rel_obj.source, rel_obj.description)
            base_key_prefixes: List[Optional[str]] = []
            base_key_prefixes.extend(referenced_law_ids)
            for base_candidate in [law_id, payload.law_id, payload.law_name]:
                if base_candidate and base_candidate not in base_key_prefixes: base_key_prefixes.append(base_candidate)
            def _match_direct_key(candidate: str) -> bool:
                nonlocal target_point_key, target_clause_key, target_article_key
                if not candidate: return False
                for base in base_key_prefixes:
                    if not base: continue
                    pattern = rf"^\s*{re.escape(base)}::(?:Điều|Article)#(\d+)(?:::(?:Khoản|Clause)#(\d+))?(?:::(?:Điểm|Point)#([A-Za-z0-9]+))?\s*$"
                    m = re.match(pattern, candidate)
                    if not m: continue
                    article_no = int(m.group(1))
                    target_article_key = f"{base}::Điều#{article_no}"
                    clause_group = m.group(2); point_group = m.group(3)
                    if clause_group is not None:
                        clause_no = int(clause_group)
                        target_clause_key = f"{target_article_key}::Khoản#{clause_no}"
                    if point_group is not None and target_clause_key:
                        psn = normalize_point_symbol(point_group)
                        if psn: target_point_key = f"{target_clause_key}::Điểm#{psn}"
                    return True
                return False
            if tgt_name and _match_direct_key(tgt_name_raw): pass
            article_match = re.search(r"(?:[Đđ]iều|Article)\s*(\d+)", tgt_name_raw)
            clause_match = re.search(r"(?:[Kk]hoản|Clause)\s*(\d+)", tgt_name_raw)
            point_match = re.search(r"(?:[Đđ]iểm|Point)\s*([A-Za-z0-9]+)", tgt_name_raw)
            try:
                if point_match and clause_match and article_match:
                    clause_no = int(clause_match.group(1)); article_no = int(article_match.group(1))
                    psn = normalize_point_symbol(point_match.group(1))
                    for base in base_key_prefixes:
                        if not base or not psn: continue
                        target_article_key = f"{base}::Điều#{article_no}"
                        target_clause_key = f"{target_article_key}::Khoản#{clause_no}"
                        target_point_key = f"{target_clause_key}::Điểm#{psn}"; break
                elif clause_match and article_match:
                    clause_no = int(clause_match.group(1)); article_no = int(article_match.group(1))
                    for base in base_key_prefixes:
                        if not base: continue
                        target_article_key = f"{base}::Điều#{article_no}"
                        target_clause_key = f"{target_article_key}::Khoản#{clause_no}"; break
                elif article_match:
                    article_no = int(article_match.group(1))
                    for base in base_key_prefixes:
                        if not base: continue
                        target_article_key = f"{base}::Điều#{article_no}"; break
            except ValueError:
                target_point_key = target_clause_key = target_article_key = None
            if not any([target_point_key, target_clause_key, target_article_key]) and tgt_name is None: return
            raw_source_name = rel_obj.source or fallback_source or payload.law_name or payload.law_id or None
            source_name = raw_source_name.strip() if isinstance(raw_source_name, str) and raw_source_name.strip() else None
            source_point_key = source_node_key if source_node_type == "POINT" else None
            source_clause_key = source_node_key if source_node_type == "CLAUSE" else None
            source_article_key = source_node_key if source_node_type == "ARTICLE" else None
            if source_name is None and not any([source_point_key, source_clause_key, source_article_key]): return
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
        # Law node
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
                for pt in cl.points:
                    psn = normalize_point_symbol(pt.point_symbol)
                    if not psn: continue
                    point_key = f"{clause_key}::Điểm#{psn}"
                    s.run(INGEST_CYPHER["point"], {
                        "clause_key": clause_key,
                        "point_key": point_key,
                        "point_symbol": pt.point_symbol or psn,
                        "point_symbol_norm": psn,
                        "content": pt.content,
                        "effective_date": pt.effective_date,
                        "expiry_date": pt.expiry_date,
                    })
                    for rel in pt.relations: ingest_relation_entry(rel, default_source, point_key, "POINT")
                for rel in cl.relations: ingest_relation_entry(rel, default_source, clause_key, "CLAUSE")

SYSTEM_INSTRUCTIONS = (
    "You are a structured information extraction assistant for Vietnamese legal texts. "
    "Emit JSON conforming to LawSchema. Keep original Vietnamese text, normalize dates (yyyy-mm-dd)."
)
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
model = OpenAIChatModel(model_name=OPENAI_MODEL, provider=OpenAIProvider(api_key=OPENAI_API_KEY))
extract_agent = Agent(model=model, output_type=LawSchema, system_prompt=SYSTEM_INSTRUCTIONS)
async def extract_schema(text: str) -> LawSchema:
    run = await extract_agent.run(text)
    return run.output

# Query helper wrappers

def ask_what_article_does(driver, law_name: str, article_number: int) -> str:
    with driver.session() as s:
        rec = s.run(GET_ARTICLE_CONTENT, {"law_name": law_name, "article_number": article_number}).single()
        if not rec: return "Không tìm thấy điều luật."
        title, content, eff = rec["title"], rec["content"], rec["law_effective"]
        out = [f"Điều {article_number} của {law_name}"]
        if title: out.append(f"(Tiêu đề: {title})")
        if content: out.append(f"nêu: {content}")
        if eff: out.append(f"(Luật có hiệu lực từ {eff}).")
        mods = [m for m in s.run(GET_MODIFIERS, {"law_name": law_name, "article_number": article_number}).data() if m["source"]]
        if mods:
            out.append("\nCác văn bản sửa đổi điều này:")
            for m in mods:
                out.append(f"- {m['source']} (hiệu lực từ {m.get('effective')})")
        return " ".join(out)

def search_articles(driver, question: str, limit: int = 5) -> List[Dict[str, Any]]:
    term = re.sub(r"\s+", " ", question).strip().lower()
    article_numbers = [int(n) for n in re.findall(r"(?i)(?:điều|article)\s*(\d+)", question)]
    law_ids = extract_law_ids(question)
    law_name_term = ""
    m = re.search(r"(?i)luật\s+([^\d,;:\n\?]+)", question)
    if m: law_name_term = re.sub(r"\s+", " ", m.group(1)).strip().lower()
    cleaned = term
    for patt in [r"(?i)(điều|article)\s*\d+", r"(?i)(khoản|clause)\s*\d+", r"(?i)(điểm|point)\s*[a-z0-9]+"]:
        cleaned = re.sub(patt, " ", cleaned)
    for lid in law_ids: cleaned = cleaned.replace(lid.lower(), " ")
    if law_name_term: cleaned = cleaned.replace(law_name_term, " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    params = {"article_numbers": article_numbers, "law_ids": [lid.upper() for lid in law_ids], "law_name_term": law_name_term, "term": cleaned, "limit": max(1, limit)}
    with driver.session() as s: return s.run(SEARCH_TOP_ARTICLES, params).data()

def search_all_nodes(driver, question: str, limit: int = 10) -> List[Dict[str, Any]]:
    term = re.sub(r"\s+", " ", question).strip().lower()
    with driver.session() as s: return s.run(SEARCH_ALL_NODES, {"term": term, "limit": max(1, limit)}).data()

def structured_candidates(driver, slots: QuerySlots) -> Dict[int, Dict[str, Any]]:
    params_common = {"law_ids": [lid.upper() for lid in slots.law_ids], "law_names": [n.lower() for n in slots.law_names], "article_numbers": slots.article_numbers, "clause_numbers": slots.clause_numbers}
    candidates: Dict[int, Dict[str, Any]] = {}
    def _register(rec: Dict[str, Any], extra: float, reason: str):
        node_id = rec["node_id"]; entry = candidates.setdefault(node_id, {"node_id": node_id, "labels": rec.get("labels", []), "identifier": rec.get("identifier"), "text": rec.get("text"), "score": 0.0, "reasons": []}); entry["score"] += extra; entry["reasons"].append(reason)
    with driver.session() as session:
        if params_common["law_ids"] or params_common["law_names"]:
            for rec in session.run(STRUCTURED_LAW_QUERY, params_common).data(): _register(rec, 8.0, "trùng luật")
            for rec in session.run(STRUCTURED_ARTICLE_QUERY, params_common).data(): _register(rec, 6.0 + (4.0 if params_common["article_numbers"] else 0.0), f"ứng viên điều {rec.get('article_number','')}")
            for rec in session.run(STRUCTURED_CLAUSE_QUERY, params_common).data():
                bonus = 5.0 + (3.0 if params_common["clause_numbers"] else 0.0) + (2.0 if params_common["article_numbers"] else 0.0)
                desc = ""; an = rec.get("article_number"); cn = rec.get("clause_number");
                if an is not None and cn is not None: desc = f" điều {an} khoản {cn}"; _register(rec, bonus, f"ứng viên khoản{desc}")
    return candidates

def fetch_node_context(driver, node_id: int, relation_limit: int = 10) -> List[Dict[str, Any]]:
    with driver.session() as s: return s.run(NODE_CONTEXT, {"node_id": node_id, "relation_limit": relation_limit}).data()

def route_query(driver, question: str, limit: int = 5) -> Dict[str, Any]:
    slots = extract_query_slots(question)
    combined = structured_candidates(driver, slots)
    for rec in search_all_nodes(driver, slots.open_text or question, limit=50):
        nid = rec.get("node_id");
        if nid is None: continue
        entry = combined.setdefault(nid, {"node_id": nid, "labels": rec.get("labels", []), "identifier": rec.get("identifier"), "text": rec.get("text"), "score": 0.0, "reasons": []})
        entry["score"] += float(rec.get("score", 0.0)); entry["reasons"].append("khớp toàn văn")
    ranked = sorted(combined.values(), key=lambda x: x["score"], reverse=True)
    for item in ranked[:limit]: item["context"] = fetch_node_context(driver, item["node_id"])
    return {"slots": slots, "results": ranked[:limit]}

def _snippet(text: Optional[str], limit: int = 180) -> str:
    if not text: return "(không có nội dung)"
    clean = re.sub(r"\s+", " ", text).strip()
    return clean if len(clean) <= limit else clean[:limit].rstrip() + "…"

SAMPLE_TEXT_1 = """LUẬT THUẾ GIÁ TRỊ GIA TĂNG\nĐiều 2. Thuế giá trị gia tăng\n1. Thuế giá trị gia tăng là thuế tính trên giá trị tăng thêm của hàng hoá, dịch vụ.\n2. Đối tượng nộp thuế là tổ chức, cá nhân sản xuất, kinh doanh hàng hoá, dịch vụ chịu thuế.\n3. Quy định chuyển tiếp áp dụng đến hết ngày 31/12/2009."""
SAMPLE_TEXT_2 = """LUẬT SỬA ĐỔI, BỔ SUNG MỘT SỐ ĐIỀU CỦA LUẬT THUẾ GTGT\nĐiều 1. Sửa đổi, bổ sung Luật số 13/2008/QH12\n1. Sửa đổi Khoản 1 Điều 2 ..."""
SAMPLE_TEXTS = [SAMPLE_TEXT_1, SAMPLE_TEXT_2]

def ensure_env() -> Neo4jConfig:
    cfg = Neo4jConfig(uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"), user=os.getenv("NEO4J_USER", "neo4j"), password=os.getenv("NEO4J_PASSWORD", "password"))
    if not os.getenv("OPENAI_API_KEY"): raise RuntimeError("OPENAI_API_KEY is not set")
    return cfg

async def demo(query: Optional[str] = None, limit: int = 5, interactive: bool = False):
    cfg = ensure_env(); driver = get_driver(cfg); init_schema(driver)
    print("\n=== Extracting JSON schema via PydanticAI + ChatGPT ===")
    law_schema: LawSchema = await extract_schema(SAMPLE_TEXT_2)
    print(json.dumps(json.loads(law_schema.model_dump_json()), ensure_ascii=False, indent=2))
    print("\n=== Ingesting into Neo4j ==="); ingest_to_neo4j(driver, law_schema)
    search_terms: List[str] = []
    if query: search_terms.append(query)
    if interactive:
        try:
            while True:
                user_q = input("\nNhập từ khóa (Enter để thoát): ").strip()
                if not user_q: break
                search_terms.append(user_q)
        except (EOFError, KeyboardInterrupt): print("\n(thoát chế độ tương tác)")
    if not search_terms: search_terms.append("LUẬT THUẾ GIÁ TRỊ GIA TĂNG Điều 2 Khoản 3")
    for q in search_terms:
        print(f"\n=== Tìm kiếm top {limit} nút cho: {q} ===")
        routed = route_query(driver, q, limit=limit)
        results = routed["results"]; slots = routed["slots"]
        if slots.is_ambiguous: print(" Thiếu thông tin luật cụ thể, kết quả có thể rộng.")
        if not results: print("(không tìm thấy kết quả)"); continue
        for idx, hit in enumerate(results, start=1):
            labels = ",".join(hit.get("labels", [])); identifier = hit.get("identifier") or ""; text = hit.get("text") or ""; snippet = _snippet(text)
            reasons = "; ".join(hit.get("reasons", []))
            print(f"{idx}. ({labels}) {'['+identifier+']' if identifier else ''}: {snippet} (score={hit['score']:.1f})")
            if reasons: print(f"   Lý do: {reasons}")
            for rel in (hit.get("context") or [])[:limit]:
                direction = "→" if rel.get("direction") == "out" else "←"; rel_type = rel.get("rel_type") or "?"; other_labels = ",".join(rel.get("labels", []))
                other_identifier = rel.get("identifier") or ""; other_text = _snippet(rel.get("text")); eff = rel.get("effective_date"); exp = rel.get("expiry_date"); desc = rel.get("description")
                meta_parts = [p for p in [f"hiệu lực {eff}" if eff else None, f"hết hiệu lực {exp}" if exp else None, desc] if p]
                meta = f" ({'; '.join(meta_parts)})" if meta_parts else ""
                print(f"      {direction} {rel_type}: ({other_labels}) {'['+other_identifier+']' if other_identifier else ''} {other_text}{meta}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo tìm kiếm nút trong đồ thị pháp luật")
    parser.add_argument("query", nargs="?", help="Từ khóa tìm kiếm ban đầu")
    parser.add_argument("--limit", type=int, default=5, help="Số kết quả tối đa")
    parser.add_argument("--interactive", action="store_true", help="Bật chế độ nhập nhiều truy vấn")
    args = parser.parse_args()
    asyncio.run(demo(query=args.query, limit=args.limit, interactive=args.interactive))
