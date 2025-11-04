from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from uraxlaw.lawgraph.models import Edge, Node
from uraxlaw.preprocess.agent_chunker import AgentChunker
from uraxlaw.preprocess.models import DocumentMetadata


# Regex patterns for extracting relationships
# Pattern to extract references in Vietnamese legal format
# Examples: "Nghị định số 24/2014/NĐ-CP", "Luật 38/2019/QH14", "Quyết định số 14/2022/QĐ-UBND"
DOC_REF_PATTERN = re.compile(
    r"(?i)(Luật|Nghị định|Thông tư|Quyết định|Nghị quyết)\s+(?:số\s+)?(\d{1,4})/(\d{4})(?:/([A-ZĐ]+-[A-Z]+))?"
)
ARTICLE_REF_PATTERN = re.compile(
    r"(?i)(?:theo|theo quy định tại|quy định tại|tại)\s+(?:điều|khoản|điểm)\s+(\d+)(?:\s+(?:khoản|điểm)\s+(\d+))?(?:\s+của\s+(?:Luật|Nghị định|Thông tư)\s+([0-9]{1,4}/[0-9]{4}))?"
)
TERM_DEF_PATTERN = re.compile(
    r'(?i)(?:Trong\s+(?:Luật|Nghị định|Thông tư)\s+này,?\s+)?(?:các\s+)?từ\s+ngữ\s+(?:dưới\s+đây\s+)?(?:được\s+)?hiểu\s+như\s+sau:\s*"([^"]+)"\s+là\s+([^"]+)'
)
GUIDES_PATTERN = re.compile(
    r"(?i)(?:hướng dẫn|thi hành)\s+(?:Luật|Nghị định|Thông tư)\s+(?:số\s+)?([0-9]{1,4}/[0-9]{4})"
)
AMENDS_PATTERN = re.compile(
    r"(?i)(?:sửa đổi|bổ sung)\s+(?:điều|khoản|điểm)\s+(\d+)(?:\s+(?:khoản|điểm)\s+(\d+))?(?:\s+của\s+(?:Luật|Nghị định|Thông tư)\s+([0-9]{1,4}/[0-9]{4}))?"
)


class Neo4jChunker:
    """
    Chunker for preparing data for Neo4j according to the new schema.
    Creates Document, Article, and Clause nodes with proper relationships.

    INPUT FORMAT REQUIREMENTS:
    ==========================
    
    1. text (str):
       - Raw Vietnamese legal document text
       - Should contain structured content with:
         * Articles (Điều X)
         * Clauses (Khoản X)
         * Points (Điểm X)
       - Example structure:
         ```
         Điều 1. Phạm vi điều chỉnh
         Luật này quy định về...
         
         Điều 2. Đối tượng áp dụng
         1. Luật này áp dụng đối với:
            a) Người nộp thuế;
            b) Cơ quan quản lý thuế;
         ```
    
    2. doc_id (str):
       - Unique document identifier
       - Format: "L-2019-38", "NĐ-2020-126", "TT-2021-15", etc.
       - Pattern: {Type}-{Year}-{Number}
       - Examples:
         * "L-2019-38" for Law 38/2019
         * "NĐ-2020-126" for Decree 126/2020
         * "TT-2021-15" for Circular 15/2021
    
    3. doc_type (str):
       - Document type classification
       - Valid values: "Law", "Decree", "Circular", "Decision", "Resolution"
       - Should match the document type in the text
    
    4. metadata (DocumentMetadata, optional):
       - DocumentMetadata object with the following fields:
         * document_id (str): Document ID (should match doc_id)
         * issuing_authority (str, optional): Issuing authority (e.g., "Quốc hội", "Chính phủ")
         * effect_date (str, optional): Effective date in ISO format (YYYY-MM-DD)
         * field (str, optional): Legal field (e.g., "Thuế", "Lao động")
         * status (str, optional): Status - "effective", "expired", "amended", "draft"
         * version (str, optional): Version number
         * source_url (str, optional): Source URL
       - Example:
         ```python
         DocumentMetadata(
             document_id="L-2019-38",
             issuing_authority="Quốc hội",
             effect_date="2019-07-01",
             field="Thuế",
             status="effective",
             source_url="https://example.com/law-38-2019"
         )
         ```

    OUTPUT FORMAT:
    ==============
    
    Returns a tuple of (nodes, edges):
    
    1. nodes (List[Dict]):
       - Each node dict contains:
         * "label" (str): Node label ("Document", "Article", "Clause", "Term", "Agency")
         * "identifier" (str): Unique identifier for the node
         * "identifier_prop" (str): Property name for the identifier ("doc_id", "article_id", etc.)
         * "properties" (Dict): Node properties
    
    2. edges (List[Dict]):
       - Each edge dict contains:
         * "source_label" (str): Source node label
         * "source_id" (str): Source node identifier
         * "source_id_prop" (str): Source identifier property name
         * "relation" (str): Relationship type (HAS_ARTICLE, HAS_CLAUSE, CITES, etc.)
         * "target_label" (str): Target node label
         * "target_id" (str): Target node identifier
         * "target_id_prop" (str): Target identifier property name
         * "properties" (Dict): Edge properties (optional)

    RELATIONSHIP TYPES EXTRACTED:
    =============================
    
    The chunker automatically extracts the following relationships:
    - HAS_ARTICLE: Document -> Article
    - HAS_CLAUSE: Article -> Clause
    - ISSUED_BY: Document -> Agency
    - CITES: Article/Clause -> Article/Clause
    - DEFINED_IN: Term -> Article
    - MENTIONED_IN: Term -> Document
    - GUIDES: Document -> Document
    - REFERENCES: Document -> Document
    - BASED_ON: Document -> Document
    - AMENDS: Article/Clause -> Article/Clause
    - SUPPLEMENTS: Document -> Document
    - REPEALS: Document -> Document
    - INTERPRETS: Document -> Document
    """

    def __init__(self, agent_chunker: Optional[AgentChunker] = None) -> None:
        """
        Initialize Neo4jChunker.

        Args:
            agent_chunker: AgentChunker instance (if None, creates one)
        """
        self._agent_chunker = agent_chunker or AgentChunker()

    async def chunk_for_neo4j(
        self,
        text: str,
        doc_id: str,
        doc_type: str,
        metadata: Optional[DocumentMetadata] = None,
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Chunk document and prepare nodes and edges for Neo4j insertion.

        Args:
            text: Raw Vietnamese legal document text with structured articles/clauses
            doc_id: Unique document identifier (e.g., "L-2019-38", "NĐ-2020-126")
            doc_type: Document type ("Law", "Decree", "Circular", etc.)
            metadata: Optional document metadata (issuing_authority, effect_date, etc.)

        Returns:
            Tuple of (nodes, edges) ready for Neo4j insertion
            nodes: List of dicts with label, identifier, identifier_prop, and properties
            edges: List of dicts with source, target, relation, and properties
        """
        # Use AgentChunker to get structured chunks
        result = await self._agent_chunker._agent.run(text)
        chunked_doc = result.output
        article_chunks = chunked_doc.chunks

        nodes: List[Dict] = []
        edges: List[Dict] = []

        # 1. Create Document node
        doc_props = {
            "doc_id": doc_id,
            "doc_type": doc_type,
        }
        if metadata:
            if metadata.issuing_authority:
                doc_props["agency"] = metadata.issuing_authority
            if metadata.effect_date:
                # Parse effect_date if it's a string
                if isinstance(metadata.effect_date, str):
                    doc_props["effective_date"] = metadata.effect_date
                    try:
                        doc_props["year"] = int(metadata.effect_date.split("-")[0])
                    except:
                        pass
                else:
                    doc_props["effective_date"] = str(metadata.effect_date)
            if metadata.field:
                doc_props["field"] = metadata.field
            if metadata.status:
                doc_props["status"] = metadata.status
            if metadata.source_url:
                doc_props["source_url"] = metadata.source_url

        # Extract title from text (first line or header)
        lines = text.strip().split("\n")
        title = next((line.strip() for line in lines if line.strip() and "LUẬT" in line.upper() or "NGHỊ ĐỊNH" in line.upper()), "")
        if title:
            doc_props["title"] = title

        nodes.append({
            "label": "Document",
            "identifier": doc_id,
            "identifier_prop": "doc_id",
            "properties": doc_props,
        })

        # 2. Create Article and Clause nodes
        seen_article_ids = set()
        seen_clause_ids = set()

        for article_chunk in article_chunks:
            # Create Article node if this is a new article
            if article_chunk.article_number is not None:
                article_id = f"{doc_id}:{article_chunk.article_number}"
                
                # Only create Article node if it's a new article
                if article_id not in seen_article_ids:
                    article_props = {
                        "article_id": article_id,
                        "number": article_chunk.article_number,
                        "text": article_chunk.content,
                    }
                    if article_chunk.title:
                        article_props["title"] = article_chunk.title

                    nodes.append({
                        "label": "Article",
                        "identifier": article_id,
                        "identifier_prop": "article_id",
                        "properties": article_props,
                    })

                    # Create Document -> Article edge
                    edges.append({
                        "source_label": "Document",
                        "source_id": doc_id,
                        "source_id_prop": "doc_id",
                        "relation": "HAS_ARTICLE",
                        "target_label": "Article",
                        "target_id": article_id,
                        "target_id_prop": "article_id",
                        "properties": {},
                    })

                    seen_article_ids.add(article_id)

            # Create Clause node if this chunk has a clause
            if article_chunk.clause_number is not None and article_chunk.article_number is not None:
                article_id = f"{doc_id}:{article_chunk.article_number}"
                clause_id = f"{article_id}:{article_chunk.clause_number}"
                
                # Only create Clause node if it's a new clause
                if clause_id not in seen_clause_ids:
                    clause_props = {
                        "clause_id": clause_id,
                        "number": str(article_chunk.clause_number),
                        "text": article_chunk.content,
                    }
                    if article_chunk.point_symbol:
                        clause_props["point_symbol"] = article_chunk.point_symbol

                    nodes.append({
                        "label": "Clause",
                        "identifier": clause_id,
                        "identifier_prop": "clause_id",
                        "properties": clause_props,
                    })

                    # Create Article -> Clause edge
                    edges.append({
                        "source_label": "Article",
                        "source_id": article_id,
                        "source_id_prop": "article_id",
                        "relation": "HAS_CLAUSE",
                        "target_label": "Clause",
                        "target_id": clause_id,
                        "target_id_prop": "clause_id",
                        "properties": {},
                    })

                    seen_clause_ids.add(clause_id)

        # 3. Extract additional relationships from text
        additional_nodes, additional_edges = self._extract_relationships(text, doc_id, doc_type, article_chunks)
        nodes.extend(additional_nodes)
        edges.extend(additional_edges)

        # 4. Create Agency node and ISSUED_BY relationship if agency is provided
        if metadata and metadata.issuing_authority:
            agency_name = metadata.issuing_authority
            agency_node = {
                "label": "Agency",
                "identifier": agency_name,
                "identifier_prop": "name",
                "properties": {"name": agency_name, "type": "government"},
            }
            nodes.append(agency_node)

            edges.append({
                "source_label": "Document",
                "source_id": doc_id,
                "source_id_prop": "doc_id",
                "relation": "ISSUED_BY",
                "target_label": "Agency",
                "target_id": agency_name,
                "target_id_prop": "name",
                "properties": {},
            })

        return nodes, edges

    def _normalize_document_reference(self, doc_type_text: str, number: str, year: str, agency_code: Optional[str] = None) -> str:
        """
        Normalize document reference to Vietnamese standard format: number/year/type-agency_code
        
        Examples:
        - "Nghị định", "24", "2014", None -> "24/2014/NĐ-CP"
        - "Luật", "38", "2019", "QH14" -> "38/2019/L-QH14"
        - "Quyết định", "14", "2022", "QĐ-UBND" -> "14/2022/QĐ-UBND"
        """
        # Map document type to type prefix and default agency code
        type_map = {
            "Luật": ("L", "QH"),
            "Nghị định": ("NĐ", "CP"),
            "Thông tư": ("TT", "BTC"),
            "Quyết định": ("QĐ", "TTg"),
            "Nghị quyết": ("NQ", "HĐND"),
        }
        
        doc_type_lower = doc_type_text.lower()
        type_prefix = None
        default_agency = None
        
        for key, (prefix, agency) in type_map.items():
            if key.lower() in doc_type_lower:
                type_prefix = prefix
                default_agency = agency
                break
        
        # If not found, default to QĐ-TTg
        if not type_prefix:
            type_prefix = "QĐ"
            default_agency = "TTg"
        
        # Use provided agency_code if available, otherwise use default
        if agency_code:
            # If agency_code is already in format "type-agency", use it
            if "-" in agency_code:
                final_type = agency_code
            else:
                # If just agency code, combine with type prefix
                final_type = f"{type_prefix}-{agency_code}"
        else:
            # Use default agency code
            final_type = f"{type_prefix}-{default_agency}"
        
        return f"{number}/{year}/{final_type}"

    def _extract_relationships(
        self, text: str, doc_id: str, doc_type: str, article_chunks: List
    ) -> Tuple[List[Dict], List[Dict]]:
        """Extract additional relationships from document text."""
        nodes: List[Dict] = []
        edges: List[Dict] = []
        seen_terms = set()

        # Extract document references (GUIDES, REFERENCES, BASED_ON)
        # Normalize references to Vietnamese standard format: number/year/type-agency_code
        doc_refs = DOC_REF_PATTERN.finditer(text)
        for match in doc_refs:
            doc_type_text = match.group(1)  # "Luật", "Nghị định", etc.
            number = match.group(2)
            year = match.group(3)
            agency_code = match.group(4)  # May be None
            
            # Normalize to Vietnamese standard format
            target_doc_id = self._normalize_document_reference(doc_type_text, number, year, agency_code)

            # Determine relationship type based on context
            context = match.group(0).lower()
            if "hướng dẫn" in context or "thi hành" in context:
                relation = "GUIDES"
            elif "căn cứ" in context:
                relation = "BASED_ON"
            else:
                relation = "REFERENCES"

            edges.append({
                "source_label": "Document",
                "source_id": doc_id,
                "source_id_prop": "doc_id",
                "relation": relation,
                "target_label": "Document",
                "target_id": target_doc_id,
                "target_id_prop": "doc_id",
                "properties": {"context": match.group(0)},
            })

        # Extract article citations (CITES)
        article_refs = ARTICLE_REF_PATTERN.finditer(text)
        for match in article_refs:
            article_num = int(match.group(1))
            clause_num = match.group(2) if match.group(2) else None
            target_doc = match.group(3) if match.group(3) else None

            # Determine source article/clause from context
            # For now, we'll create citations from current document's articles
            source_article_id = f"{doc_id}:{article_num}"
            if target_doc:
                target_doc_id = f"Law_{target_doc}"
                target_article_id = f"{target_doc_id}:{article_num}"
            else:
                target_article_id = source_article_id

            if clause_num:
                source_clause_id = f"{source_article_id}:{clause_num}"
                target_clause_id = f"{target_article_id}:{clause_num}" if target_doc else source_clause_id
                edges.append({
                    "source_label": "Clause",
                    "source_id": source_clause_id,
                    "source_id_prop": "clause_id",
                    "relation": "CITES",
                    "target_label": "Clause",
                    "target_id": target_clause_id,
                    "target_id_prop": "clause_id",
                    "properties": {},
                })
            else:
                edges.append({
                    "source_label": "Article",
                    "source_id": source_article_id,
                    "source_id_prop": "article_id",
                    "relation": "CITES",
                    "target_label": "Article",
                    "target_id": target_article_id,
                    "target_id_prop": "article_id",
                    "properties": {},
                })

        # Extract term definitions (DEFINED_IN)
        term_defs = TERM_DEF_PATTERN.finditer(text)
        for match in term_defs:
            term_name = match.group(1).strip()
            term_definition = match.group(2).strip()
            normalized_term = term_name.lower().replace(" ", "_")

            # Avoid duplicates
            if normalized_term in seen_terms:
                continue
            seen_terms.add(normalized_term)

            # Find which article this definition is in
            for article_chunk in article_chunks:
                if article_chunk.article_number is not None and term_definition in article_chunk.content:
                    article_id = f"{doc_id}:{article_chunk.article_number}"
                    
                    # Create Term node
                    term_node = {
                        "label": "Term",
                        "identifier": normalized_term,
                        "identifier_prop": "normalized_name",
                        "properties": {
                            "name": term_name,
                            "normalized_name": normalized_term,
                            "definition_text": term_definition,
                        },
                    }
                    nodes.append(term_node)
                    
                    edges.append({
                        "source_label": "Term",
                        "source_id": normalized_term,
                        "source_id_prop": "normalized_name",
                        "relation": "DEFINED_IN",
                        "target_label": "Article",
                        "target_id": article_id,
                        "target_id_prop": "article_id",
                        "properties": {},
                    })
                    break

        # Extract AMENDS relationships
        amends_refs = AMENDS_PATTERN.finditer(text)
        for match in amends_refs:
            article_num = int(match.group(1))
            clause_num = match.group(2) if match.group(2) else None
            target_doc = match.group(3) if match.group(3) else None

            source_article_id = f"{doc_id}:{article_num}"
            if target_doc:
                target_doc_id = f"Law_{target_doc}"
                target_article_id = f"{target_doc_id}:{article_num}"
                
                if clause_num:
                    target_clause_id = f"{target_article_id}:{clause_num}"
                    source_clause_id = f"{source_article_id}:{clause_num}"
                    edges.append({
                        "source_label": "Clause",
                        "source_id": source_clause_id,
                        "source_id_prop": "clause_id",
                        "relation": "AMENDS",
                        "target_label": "Clause",
                        "target_id": target_clause_id,
                        "target_id_prop": "clause_id",
                        "properties": {},
                    })
                else:
                    edges.append({
                        "source_label": "Article",
                        "source_id": source_article_id,
                        "source_id_prop": "article_id",
                        "relation": "AMENDS",
                        "target_label": "Article",
                        "target_id": target_article_id,
                        "target_id_prop": "article_id",
                        "properties": {},
                    })

        # Extract SUPPLEMENTS relationships (when document supplements another)
        supplements_pattern = re.compile(r"(?i)(?:bổ sung|quy định bổ sung)\s+(?:điều|khoản)\s+(\d+)(?:\s+của\s+(?:Luật|Nghị định|Thông tư)\s+([0-9]{1,4}/[0-9]{4}))?")
        supplements_refs = supplements_pattern.finditer(text)
        for match in supplements_refs:
            article_num = int(match.group(1))
            target_doc = match.group(2) if match.group(2) else None

            source_article_id = f"{doc_id}:{article_num}"
            if target_doc:
                target_doc_id = f"Law_{target_doc}"
                target_article_id = f"{target_doc_id}:{article_num}"
                
                edges.append({
                    "source_label": "Document",
                    "source_id": doc_id,
                    "source_id_prop": "doc_id",
                    "relation": "SUPPLEMENTS",
                    "target_label": "Document",
                    "target_id": target_doc_id,
                    "target_id_prop": "doc_id",
                    "properties": {"article": article_num},
                })

        # Extract REPEALS relationships
        repeals_pattern = re.compile(r"(?i)(?:bãi bỏ|hủy bỏ|thay thế)\s+(?:Luật|Nghị định|Thông tư)\s+(?:số\s+)?([0-9]{1,4}/[0-9]{4})")
        repeals_refs = repeals_pattern.finditer(text)
        for match in repeals_refs:
            target_doc_number = match.group(1)
            target_doc_id = f"Law_{target_doc_number}"
            
            edges.append({
                "source_label": "Document",
                "source_id": doc_id,
                "source_id_prop": "doc_id",
                "relation": "REPEALS",
                "target_label": "Document",
                "target_id": target_doc_id,
                "target_id_prop": "doc_id",
                "properties": {},
            })

        # Extract INTERPRETS relationships (when document interprets another)
        interprets_pattern = re.compile(r"(?i)(?:giải thích|hướng dẫn thi hành)\s+(?:Luật|Nghị định|Thông tư)\s+(?:số\s+)?([0-9]{1,4}/[0-9]{4})")
        interprets_refs = interprets_pattern.finditer(text)
        for match in interprets_refs:
            target_doc_number = match.group(1)
            target_doc_id = f"Law_{target_doc_number}"
            
            edges.append({
                "source_label": "Document",
                "source_id": doc_id,
                "source_id_prop": "doc_id",
                "relation": "INTERPRETS",
                "target_label": "Document",
                "target_id": target_doc_id,
                "target_id_prop": "doc_id",
                "properties": {},
            })

        # Extract MENTIONED_IN relationships (terms/concepts mentioned in documents)
        # This is a simplified version - can be enhanced with NLP
        common_legal_terms = ["người nộp thuế", "cơ quan quản lý thuế", "tổ chức", "cá nhân"]
        for term in common_legal_terms:
            if term.lower() in text.lower():
                normalized_term = term.lower().replace(" ", "_")
                if normalized_term not in seen_terms:
                    term_node = {
                        "label": "Term",
                        "identifier": normalized_term,
                        "identifier_prop": "normalized_name",
                        "properties": {
                            "name": term,
                            "normalized_name": normalized_term,
                        },
                    }
                    nodes.append(term_node)
                    seen_terms.add(normalized_term)
                    
                    edges.append({
                        "source_label": "Term",
                        "source_id": normalized_term,
                        "source_id_prop": "normalized_name",
                        "relation": "MENTIONED_IN",
                        "target_label": "Document",
                        "target_id": doc_id,
                        "target_id_prop": "doc_id",
                        "properties": {},
                    })

        return nodes, edges

