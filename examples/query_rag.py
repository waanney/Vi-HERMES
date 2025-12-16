"""
Query RAG System with Chainlit: Query from Milvus (KB) and Neo4j (KG) and generate answer with GPT-4o

This script demonstrates:
1. Query from Milvus (Knowledge Base - KB) for vector search
2. Query from Neo4j (Knowledge Graph - KG) for graph traversal
3. Combine results and generate answer with GPT-4o
4. Show clear source attribution (from KB or KG)
5. Interactive UI with Chainlit
"""

from __future__ import annotations

from typing import List, Optional

# Conditional chainlit import - only needed for chainlit UI functions
try:
    import chainlit as cl
    CHAINLIT_AVAILABLE = True
except (ImportError, ValueError, Exception):
    # Catch all exceptions including config errors
    CHAINLIT_AVAILABLE = False
    cl = None  # type: ignore

from dotenv import load_dotenv

from uraxlaw.Agents.engine import LLMClient
from uraxlaw.config.settings import get_settings
from uraxlaw.lawgraph.neo4j_client import Neo4jClient
from uraxlaw.lawrag.milvus_client import MilvusClient

load_dotenv()

# Global clients (initialized once)
_milvus_client: Optional[MilvusClient] = None
_neo4j_client: Optional[Neo4jClient] = None
_llm_client: Optional[LLMClient] = None
_settings = None


def setup_milvus_client(settings) -> MilvusClient:
    """Setup Milvus client with embedder and load collection."""
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("intfloat/multilingual-e5-large")
        dense_dim = model.get_sentence_embedding_dimension()

        def embedder(text: str):
            vec = model.encode([text], normalize_embeddings=True)[0]
            return vec.tolist()

        # Create MilvusClient with correct parameters
        milvus_client = MilvusClient(
            host=settings.milvus_host,
            port=settings.milvus_port,
            collection=settings.milvus_collection,
            dense_dim=dense_dim,
            embedder=embedder,
        )

        return milvus_client
    except Exception as e:
        raise RuntimeError(f"Failed to setup Milvus client: {e}")


def setup_neo4j_client(settings) -> Neo4jClient:
    """Setup Neo4j client."""
    neo4j = Neo4jClient(
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password,
    )
    return neo4j


def query_milvus(milvus_client: MilvusClient, query: str, top_k: int = 5) -> List[dict]:
    """Query Milvus (KB) and return results."""
    results = milvus_client.search(query=query, top_k=top_k)
    
    kb_results = []
    for result in results:
        chunk = result.chunk
        # Extract article_id from chunk.id (format: doc_id:article_number:clause_number)
        article_id = chunk.id
        doc_id = chunk.document_id
        
        # Parse article_number and clause_number from article_id
        article_number = None
        clause_number = None
        if ":" in article_id:
            parts = article_id.split(":")
            doc_id_from_id = parts[0]  # First part is doc_id
            if len(parts) >= 2:
                try:
                    article_number = int(parts[1])
                except ValueError:
                    pass
            if len(parts) >= 3:
                clause_number = parts[2]
        
        # Build article_id for Neo4j query (format: doc_id:article_number or doc_id:article_number:clause_number)
        neo4j_article_id = None
        if article_number is not None:
            neo4j_article_id_parts = [doc_id, str(article_number)]
            if clause_number:
                neo4j_article_id_parts.append(str(clause_number))
            neo4j_article_id = ":".join(neo4j_article_id_parts)
        
        # Get title from result object (stored in MilvusClient.search)
        title = getattr(result, "title", "") or ""
        
        kb_results.append({
            "source": "KB",  # Knowledge Base (Milvus)
            "article_id": neo4j_article_id,  # Use this to query Neo4j
            "article_id_original": article_id,  # Original article_id from chunk.id (format: doc_id:row_id, e.g., "1/QĐ-XPHC:5962-3912-9440")
            "doc_id": doc_id,
            "text": chunk.text,
            "title": title,
            "score": result.score,
            "article_number": article_number,
            "clause_number": clause_number,
        })
    
    return kb_results


def query_neo4j(neo4j: Neo4jClient, query_text: str, kb_results: List[dict] = None, limit: int = 5) -> List[dict]:
    """
    Query Neo4j (KG) using article_id and doc_id from KB results.
    
    Args:
        neo4j: Neo4j client
        query_text: Query text (for fallback text search)
        kb_results: Results from Milvus (KB) to extract article_id and doc_id
        limit: Maximum number of results
    
    Returns:
        List of KG results
    """
    with neo4j._driver.session() as session:
        kg_results = []
        seen_ids = set()
        
        # First: Query Neo4j using article_id and doc_id from KB results
        if kb_results:
            for kb_result in kb_results:
                doc_id = kb_result.get("doc_id")
                article_id = kb_result.get("article_id")
                
                # Query Document node by doc_id
                if doc_id:
                    doc_query = """
                    MATCH (d:Document {doc_id: $doc_id})
                    RETURN 'Document' as label,
                           d.doc_id as doc_id,
                           null as article_id,
                           null as clause_id,
                           d.text as text,
                           d.title as title,
                           d.number as number,
                           1.0 as score
                    LIMIT 1
                    """
                    result = session.run(doc_query, {"doc_id": doc_id})
                    for record in result:
                        doc_id_val = record.get("doc_id")
                        if doc_id_val and doc_id_val not in seen_ids:
                            seen_ids.add(doc_id_val)
                            kg_results.append({
                                "source": "KG",
                                "label": record["label"],
                                "doc_id": doc_id_val,
                                "article_id": None,
                                "clause_id": None,
                                "text": record.get("text", ""),
                                "title": record.get("title", ""),
                                "number": record.get("number"),
                                "score": record.get("score", 0.0),
                            })
                    
                    # Query relationships for this document
                    # Query all relationships without specifying types to avoid errors if types don't exist
                    rel_query = """
                    MATCH (d:Document {doc_id: $doc_id})-[r]->(related:Document)
                    WHERE type(r) IN ['AMENDS', 'REPEALS', 'SUPPLEMENTS', 'CITES', 'REFERENCES', 'BASED_ON', 'GUIDES', 'HAS_ARTICLE', 'HAS_CLAUSE', 'ISSUED_BY', 'DEFINED_IN', 'MENTIONED_IN', 'VERSION_OF', 'HAS_KEYWORD', 'APPLIES_TO', 'REFERENCED_BY', 'INTERPRETS']
                    RETURN 'Document' as label,
                           related.doc_id as doc_id,
                           null as article_id,
                           null as clause_id,
                           related.text as text,
                           related.title as title,
                           related.number as number,
                           type(r) as relation_type,
                           0.9 as score
                    LIMIT 10
                    """
                    try:
                        result = session.run(rel_query, {"doc_id": doc_id})
                        for record in result:
                            related_doc_id = record.get("doc_id")
                            relation_type = record.get("relation_type")
                            if related_doc_id and related_doc_id not in seen_ids:
                                seen_ids.add(related_doc_id)
                                kg_results.append({
                                    "source": "KG",
                                    "label": record["label"],
                                    "doc_id": related_doc_id,
                                    "article_id": None,
                                    "clause_id": None,
                                    "text": record.get("text", ""),
                                    "title": record.get("title", ""),
                                    "number": record.get("number"),
                                    "relation_type": relation_type,
                                    "score": record.get("score", 0.0),
                                })
                    except Exception as e:
                        # If query fails, try querying all relationships without filter
                        try:
                            rel_query_all = """
                            MATCH (d:Document {doc_id: $doc_id})-[r]->(related:Document)
                            RETURN 'Document' as label,
                                   related.doc_id as doc_id,
                                   null as article_id,
                                   null as clause_id,
                                   related.text as text,
                                   related.title as title,
                                   related.number as number,
                                   type(r) as relation_type,
                                   0.9 as score
                            LIMIT 10
                            """
                            result = session.run(rel_query_all, {"doc_id": doc_id})
                            for record in result:
                                related_doc_id = record.get("doc_id")
                                relation_type = record.get("relation_type")
                                if related_doc_id and related_doc_id not in seen_ids:
                                    seen_ids.add(related_doc_id)
                                    kg_results.append({
                                        "source": "KG",
                                        "label": record["label"],
                                        "doc_id": related_doc_id,
                                        "article_id": None,
                                        "clause_id": None,
                                        "text": record.get("text", ""),
                                        "title": record.get("title", ""),
                                        "number": record.get("number"),
                                        "relation_type": relation_type,
                                        "score": record.get("score", 0.0),
                                    })
                        except Exception:
                            pass  # Skip if still fails
                    
                    # Also query reverse relationships
                    reverse_rel_query = """
                    MATCH (related:Document)-[r]->(d:Document {doc_id: $doc_id})
                    WHERE type(r) IN ['AMENDS', 'REPEALS', 'SUPPLEMENTS', 'CITES', 'REFERENCES', 'BASED_ON', 'GUIDES', 'HAS_ARTICLE', 'HAS_CLAUSE', 'ISSUED_BY', 'DEFINED_IN', 'MENTIONED_IN', 'VERSION_OF', 'HAS_KEYWORD', 'APPLIES_TO', 'REFERENCED_BY', 'INTERPRETS']
                    RETURN 'Document' as label,
                           related.doc_id as doc_id,
                           null as article_id,
                           null as clause_id,
                           related.text as text,
                           related.title as title,
                           related.number as number,
                           type(r) as relation_type,
                           0.9 as score
                    LIMIT 10
                    """
                    try:
                        result = session.run(reverse_rel_query, {"doc_id": doc_id})
                        for record in result:
                            related_doc_id = record.get("doc_id")
                            relation_type = record.get("relation_type")
                            if related_doc_id and related_doc_id not in seen_ids:
                                seen_ids.add(related_doc_id)
                                kg_results.append({
                                    "source": "KG",
                                    "label": record["label"],
                                    "doc_id": related_doc_id,
                                    "article_id": None,
                                    "clause_id": None,
                                    "text": record.get("text", ""),
                                    "title": record.get("title", ""),
                                    "number": record.get("number"),
                                    "relation_type": relation_type,
                                    "score": record.get("score", 0.0),
                                })
                    except Exception as e:
                        # If query fails, try querying all relationships without filter
                        try:
                            reverse_rel_query_all = """
                            MATCH (related:Document)-[r]->(d:Document {doc_id: $doc_id})
                            RETURN 'Document' as label,
                                   related.doc_id as doc_id,
                                   null as article_id,
                                   null as clause_id,
                                   related.text as text,
                                   related.title as title,
                                   related.number as number,
                                   type(r) as relation_type,
                                   0.9 as score
                            LIMIT 10
                            """
                            result = session.run(reverse_rel_query_all, {"doc_id": doc_id})
                            for record in result:
                                related_doc_id = record.get("doc_id")
                                relation_type = record.get("relation_type")
                                if related_doc_id and related_doc_id not in seen_ids:
                                    seen_ids.add(related_doc_id)
                                    kg_results.append({
                                        "source": "KG",
                                        "label": record["label"],
                                        "doc_id": related_doc_id,
                                        "article_id": None,
                                        "clause_id": None,
                                        "text": record.get("text", ""),
                                        "title": record.get("title", ""),
                                        "number": record.get("number"),
                                        "relation_type": relation_type,
                                        "score": record.get("score", 0.0),
                                    })
                        except Exception:
                            pass  # Skip if still fails
                
                # Query Article node by article_id (format: doc_id:article_number)
                if article_id and ":" in article_id:
                    # article_id format: doc_id:article_number or doc_id:article_number:clause_number
                    article_query = """
                    MATCH (a:Article {article_id: $article_id})
                    RETURN 'Article' as label,
                           a.article_id as doc_id,
                           a.article_id as article_id,
                           null as clause_id,
                           a.text as text,
                           a.title as title,
                           a.number as number,
                           1.0 as score
                    LIMIT 1
                    """
                    result = session.run(article_query, {"article_id": article_id})
                    for record in result:
                        article_id_val = record.get("article_id")
                        if article_id_val and article_id_val not in seen_ids:
                            seen_ids.add(article_id_val)
                            kg_results.append({
                                "source": "KG",
                                "label": record["label"],
                                "doc_id": article_id_val.split(":")[0] if ":" in article_id_val else None,
                                "article_id": article_id_val,
                                "clause_id": None,
                                "text": record.get("text", ""),
                                "title": record.get("title", ""),
                                "number": record.get("number"),
                                "score": record.get("score", 0.0),
                            })
                    
                    # Expand related nodes (1-hop traversal)
                    expand_query = """
                    MATCH (a:Article {article_id: $article_id})-[r]-(related)
                    RETURN labels(related)[0] as label,
                           related.doc_id as doc_id,
                           related.article_id as article_id,
                           related.clause_id as clause_id,
                           related.text as text,
                           related.title as title,
                           related.number as number,
                           type(r) as relation_type,
                           0.8 as score
                    LIMIT 5
                    """
                    result = session.run(expand_query, {"article_id": article_id})
                    for record in result:
                        related_id = record.get("doc_id") or record.get("article_id") or record.get("clause_id")
                        if related_id and related_id not in seen_ids:
                            seen_ids.add(related_id)
                            kg_results.append({
                                "source": "KG",
                                "label": record["label"],
                                "doc_id": record.get("doc_id"),
                                "article_id": record.get("article_id"),
                                "clause_id": record.get("clause_id"),
                                "text": record.get("text", ""),
                                "title": record.get("title", ""),
                                "number": record.get("number"),
                                "relation_type": record.get("relation_type"),
                                "score": record.get("score", 0.0),
                            })
                
                # Query Clause nodes if article_id has clause_number
                if article_id and article_id.count(":") >= 2:
                    # Format: doc_id:article_number:clause_number
                    clause_id = article_id
                    clause_query = """
                    MATCH (c:Clause {clause_id: $clause_id})
                    RETURN 'Clause' as label,
                           c.clause_id as doc_id,
                           null as article_id,
                           c.clause_id as clause_id,
                           c.text as text,
                           null as title,
                           c.number as number,
                           1.0 as score
                    LIMIT 1
                    """
                    result = session.run(clause_query, {"clause_id": clause_id})
                    for record in result:
                        clause_id_val = record.get("clause_id")
                        if clause_id_val and clause_id_val not in seen_ids:
                            seen_ids.add(clause_id_val)
                            kg_results.append({
                                "source": "KG",
                                "label": record["label"],
                                "doc_id": clause_id_val.split(":")[0] if ":" in clause_id_val else None,
                                "article_id": ":".join(clause_id_val.split(":")[:2]) if ":" in clause_id_val else None,
                                "clause_id": clause_id_val,
                                "text": record.get("text", ""),
                                "title": None,
                                "number": record.get("number"),
                                "score": record.get("score", 0.0),
                            })
        
        # Fallback: Use text search if no KB results or not enough results
        if len(kg_results) < limit:
            # Fallback: Use text search with CONTAINS
            # Split query into keywords
            keywords = query_text.lower().split()
            
            # Search in Document nodes
            doc_query = """
            MATCH (d:Document)
            WHERE d.title CONTAINS $keyword OR d.text CONTAINS $keyword
            RETURN 'Document' as label,
                   d.doc_id as doc_id,
                   null as article_id,
                   null as clause_id,
                   d.text as text,
                   d.title as title,
                   d.number as number,
                   1.0 as score
            LIMIT $limit
            """
            
            # Search in Article nodes
            article_query = """
            MATCH (a:Article)
            WHERE a.text CONTAINS $keyword OR a.title CONTAINS $keyword
            RETURN 'Article' as label,
                   a.article_id as doc_id,
                   a.article_id as article_id,
                   null as clause_id,
                   a.text as text,
                   a.title as title,
                   a.number as number,
                   1.0 as score
            LIMIT $limit
            """
            
            # Search in Clause nodes
            clause_query = """
            MATCH (c:Clause)
            WHERE c.text CONTAINS $keyword
            RETURN 'Clause' as label,
                   c.clause_id as doc_id,
                   null as article_id,
                   c.clause_id as clause_id,
                   c.text as text,
                   null as title,
                   c.number as number,
                   1.0 as score
            LIMIT $limit
            """
            
            seen_ids = set()
            for keyword in keywords[:3]:  # Limit to first 3 keywords
                # Query documents
                for query in [doc_query, article_query, clause_query]:
                    result = session.run(query, {"keyword": keyword, "limit": limit})
                    for record in result:
                        doc_id = record.get("doc_id") or record.get("article_id") or record.get("clause_id")
                        if doc_id and doc_id not in seen_ids:
                            seen_ids.add(doc_id)
                            kg_results.append({
                                "source": "KG",
                                "label": record["label"],
                                "doc_id": doc_id,
                                "article_id": record.get("article_id"),
                                "clause_id": record.get("clause_id"),
                                "text": record.get("text", ""),
                                "title": record.get("title", ""),
                                "number": record.get("number"),
                                "score": record.get("score", 0.0),
                            })
                            
                            if len(kg_results) >= limit:
                                break
                    
                    if len(kg_results) >= limit:
                        break
                
                if len(kg_results) >= limit:
                    break
    
    return kg_results[:limit]


def build_enhanced_prompt(query: str, kb_results: List[dict], kg_results: List[dict], context: str = "") -> str:
    """Build enhanced prompt with clear source attribution."""
    context_parts = []
    
    # Add original context from CSV if provided
    if context:
        context_parts.append("=== ORIGINAL CONTEXT FROM DOCUMENT ===\n")
        context_parts.append(f"{context}\n")
    
    # Add KB results (Knowledge Base - Milvus)
    if kb_results:
        context_parts.append("=== SOURCES FROM KNOWLEDGE BASE (KB - Milvus) ===\n")
        for i, result in enumerate(kb_results, 1):
            source_info = f"[KB-{i}]"
            if result.get("doc_id"):
                source_info += f" Doc: {result['doc_id']}"
            if result.get("article_number"):
                source_info += f" | Article: {result['article_number']}"
            if result.get("clause_number"):
                source_info += f" | Clause: {result['clause_number']}"
            if result.get("title"):
                source_info += f" | Title: {result['title']}"
            source_info += f" | Score: {result.get('score', 0):.4f}"
            
            context_parts.append(f"{source_info}\n{result['text']}\n")
    
    # Add KG results (Knowledge Graph - Neo4j)
    if kg_results:
        context_parts.append("\n=== SOURCES FROM KNOWLEDGE GRAPH (KG - Neo4j) ===\n")
        for i, result in enumerate(kg_results, 1):
            source_info = f"[KG-{i}]"
            source_info += f" Type: {result.get('label', 'Unknown')}"
            if result.get("doc_id"):
                source_info += f" | Doc: {result['doc_id']}"
            if result.get("article_id"):
                source_info += f" | Article: {result['article_id']}"
            if result.get("clause_id"):
                source_info += f" | Clause: {result['clause_id']}"
            if result.get("number"):
                source_info += f" | No.: {result['number']}"
            if result.get("title"):
                source_info += f" | Title: {result['title']}"
            # Add relationship type if available
            if result.get("relation_type"):
                relation_type = result["relation_type"]
                source_info += f" | Relation: {relation_type}"
            source_info += f" | Score: {result.get('score', 0):.4f}"
            
            context_parts.append(f"{source_info}\n{result.get('text', '')}\n")
    
    context_str = "\n".join(context_parts)
    
    # Build prompt with appropriate context description
    if context:
        context_description = "Below is the original document context and the sources from the Knowledge Base (KB) and Knowledge Graph (KG):"
    else:
        context_description = "Below are the sources from the Knowledge Base (KB) and Knowledge Graph (KG):"
    
    prompt = f"""You are a Vietnamese legal expert. {context_description}

{context_str}

Question: {query}

Instructions:
1. Answer the question using the sources above (prioritize the original context if provided).
2. **IMPORTANT**: Only cite references from KB and KG, do not copy the full text. Use the format [KB-X] or [KG-X] where X is the index.
3. If both KB and KG contain relevant information, highlight differences or complementary details.
4. Specify the law, article, clause, or paragraph mentioned in the reference when applicable.
5. **IMPORTANT**: If the question is about a specific document (e.g., "What is 14/2022/QĐ-UBND about?"), you must:
   - Summarize the main content of that document.
   - Automatically list related documents based on Knowledge Graph relationships (AMENDS, REPEALS, SUPPLEMENTS, CITES/REFERENCES, GUIDES, etc.).
   - Explain the relationship type if it is provided in the KG results.
6. Respond in Vietnamese clearly and accurately. Only print references as [KB-X] or [KG-X]; do not copy entire source texts."""
    
    return prompt


async def get_clients():
    """Get or initialize clients (singleton pattern)."""
    global _milvus_client, _neo4j_client, _llm_client, _settings
    
    if not CHAINLIT_AVAILABLE:
        raise RuntimeError("Chainlit is not available. This function requires chainlit.")
    
    if _settings is None:
        _settings = get_settings()
    
    if _milvus_client is None:
        _milvus_client = setup_milvus_client(_settings)
        # Ensure collection is loaded
        try:
            client = _milvus_client._ensure_client()
            if client.has_collection(collection_name=_settings.milvus_collection):
                # Check if collection is already loaded
                try:
                    client.load_collection(collection_name=_settings.milvus_collection)
                    await cl.Message(content=f"✅ Collection '{_settings.milvus_collection}' loaded into memory").send()
                except Exception as e:
                    # Collection might already be loaded
                    if "already loaded" not in str(e).lower():
                        await cl.Message(content=f"⚠️ Could not load collection: {e}").send()
        except Exception as e:
            await cl.Message(content=f"⚠️ Error checking collection: {e}").send()
    
    if _neo4j_client is None:
        _neo4j_client = setup_neo4j_client(_settings)
    
    if _llm_client is None:
        _llm_client = LLMClient(model="gpt-4o")
    
    return _milvus_client, _neo4j_client, _llm_client


async def query_rag_system(query: str, top_k: int = 5):
    """Query RAG system and generate answer with Chainlit UI."""
    if not CHAINLIT_AVAILABLE:
        raise RuntimeError("Chainlit is not available. This function requires chainlit.")
    
    # Get clients
    milvus_client, neo4j, llm_client = await get_clients()
    
    # Show query
    await cl.Message(content=f"**Câu hỏi:** {query}").send()
    
    # Query KB (Milvus)
    async with cl.Step(name="Querying Knowledge Base (KB - Milvus)", type="tool") as step:
        try:
            kb_results = query_milvus(milvus_client, query, top_k=top_k)
            step.output = f"✅ Found {len(kb_results)} results from KB"
            if kb_results:
                result_list = []
                for i, result in enumerate(kb_results[:3], 1):
                    result_list.append(f"{i}. Doc: {result.get('doc_id')}, Score: {result.get('score', 0):.4f}")
                step.output += "\n" + "\n".join(result_list)
        except Exception as e:
            step.output = f"⚠️ KB query error: {e}"
            kb_results = []
    
    # Query KG (Neo4j)
    async with cl.Step(name="Querying Knowledge Graph (KG - Neo4j)", type="tool") as step:
        try:
            kg_results = query_neo4j(neo4j, query, kb_results=kb_results, limit=top_k)
            step.output = f"✅ Found {len(kg_results)} results from KG"
            if kg_results:
                result_list = []
                for i, result in enumerate(kg_results[:3], 1):
                    label = result.get('label', 'Unknown')
                    doc_id = result.get('doc_id', 'N/A')
                    article_id = result.get('article_id', '')
                    clause_id = result.get('clause_id', '')
                    info = f"{i}. {label}: {doc_id}"
                    if article_id:
                        info += f" | Article: {article_id}"
                    if clause_id:
                        info += f" | Clause: {clause_id}"
                    info += f" | Score: {result.get('score', 0):.4f}"
                    result_list.append(info)
                step.output += "\n" + "\n".join(result_list)
        except Exception as e:
            step.output = f"⚠️ KG query error: {e}"
            import traceback
            step.output += "\n" + traceback.format_exc()
            kg_results = []
    
    # Generate answer
    if not kb_results and not kg_results:
        await cl.Message(content="❌ No results found from either KB or KG").send()
        return
    
    async with cl.Step(name="Generating answer with GPT-4o", type="llm") as step:
        try:
            prompt = build_enhanced_prompt(query, kb_results, kg_results)
            answer = llm_client.complete(prompt)
            step.output = answer
        except Exception as e:
            step.output = f"❌ Answer generation failed: {e}"
            import traceback
            step.output += "\n" + traceback.format_exc()
            return
    
    # Show answer
    await cl.Message(content=answer).send()
    
    # Show sources
    sources_content = "## 📚 Sources\n\n"
    
    if kb_results:
        sources_content += "### Knowledge Base (KB - Milvus)\n\n"
        for i, result in enumerate(kb_results, 1):
            sources_content += f"**[KB-{i}]** {result.get('doc_id')}\n"
            if result.get('article_number'):
                sources_content += f"- Điều: {result['article_number']}\n"
            if result.get('clause_number'):
                sources_content += f"- Khoản: {result['clause_number']}\n"
            sources_content += f"- Score: {result.get('score', 0):.4f}\n\n"
    
    if kg_results:
        sources_content += "### Knowledge Graph (KG - Neo4j)\n\n"
        for i, result in enumerate(kg_results, 1):
            sources_content += f"**[KG-{i}]** {result.get('label')}: {result.get('doc_id')}\n"
            if result.get('number'):
                sources_content += f"- Số: {result['number']}\n"
            sources_content += f"- Score: {result.get('score', 0):.4f}\n\n"
    
    await cl.Message(content=sources_content).send()


if CHAINLIT_AVAILABLE:
    @cl.on_chat_start
    async def start():
        """Initialize the chat session."""
        await cl.Message(
            content="Xin chào! Tôi là trợ lý pháp lý Việt Nam. Tôi có thể trả lời các câu hỏi về pháp luật dựa trên Knowledge Base (KB - Milvus) và Knowledge Graph (KG - Neo4j).\n\nHãy đặt câu hỏi của bạn!"
        ).send()


    @cl.on_message
    async def main(message: cl.Message):
        """Handle user messages."""
        query = message.content
        await query_rag_system(query)

