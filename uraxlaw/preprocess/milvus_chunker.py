from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional

from uraxlaw.preprocess.agent_chunker import AgentChunker
from uraxlaw.preprocess.models import DocumentMetadata


class MilvusChunker:
    """
    Chunker for preparing data for Milvus according to the new schema.
    Converts AgentChunker output to Milvus format with all required fields.
    """

    def __init__(
        self,
        agent_chunker: Optional[AgentChunker] = None,
        embedder: Optional[Callable[[str], List[float]]] = None,
        sparse_encoder: Optional[Callable[[str], Dict[int, float]]] = None,
    ) -> None:
        """
        Initialize MilvusChunker.

        Args:
            agent_chunker: AgentChunker instance (if None, creates one)
            embedder: Embedding function for dense_vector (if None, embeddings won't be generated)
            sparse_encoder: Sparse encoder function for sparse_vector (if None, sparse vectors won't be generated)
        """
        self._agent_chunker = agent_chunker or AgentChunker()
        self._embedder = embedder
        self._sparse_encoder = sparse_encoder
        self._tfidf_vectorizer = None

    async def chunk_for_milvus(
        self,
        text: str,
        doc_id: str,
        doc_type: str,
        metadata: Optional[DocumentMetadata] = None,
    ) -> List[Dict]:
        """
        Chunk document and prepare data for Milvus insertion.

        Args:
            text: Document text
            doc_id: Document ID (e.g., "L-2013-43")
            doc_type: Document type (Law, Decree, Circular, etc.)
            metadata: Optional document metadata

        Returns:
            List of dictionaries ready for Milvus insertion
        """
        # Use AgentChunker to get structured chunks
        result = await self._agent_chunker._agent.run(text)
        chunked_doc = result.output
        article_chunks = chunked_doc.chunks
        
        # Build agent_chunks for reference
        agent_chunks = []
        for idx, article_chunk in enumerate(article_chunks):
            # Build chunk_id: Article_X_Clause_Y_Point_Z
            parts = []
            if article_chunk.article_number is not None:
                parts.append(f"Article_{article_chunk.article_number}")
            if article_chunk.clause_number is not None:
                parts.append(f"Clause_{article_chunk.clause_number}")
            if article_chunk.point_symbol:
                parts.append(f"Point_{article_chunk.point_symbol}")
            
            # Build content with title if available
            content = f"{article_chunk.title}\n{article_chunk.content}" if article_chunk.title else article_chunk.content
            agent_chunks.append({"id": "_".join(parts) if parts else f"chunk_{idx}", "text": content})

        milvus_data = []
        current_time = int(time.time())

        # Extract metadata fields
        year = None
        agency = None
        status = "active"
        effective_date = None
        source_url = ""

        if metadata:
            # Parse effect_date if it's a string
            if metadata.effect_date:
                if isinstance(metadata.effect_date, str):
                    # Try to extract year from date string (YYYY-MM-DD or similar)
                    try:
                        year = int(metadata.effect_date.split("-")[0])
                        effective_date = metadata.effect_date
                    except:
                        effective_date = metadata.effect_date
                else:
                    effective_date = str(metadata.effect_date)
                    year = int(str(metadata.effect_date).split("-")[0]) if "-" in str(metadata.effect_date) else None
            
            agency = metadata.issuing_authority or ""
            status = metadata.status or "active"
            source_url = metadata.source_url or ""

        # Build sparse encoder if not provided but needed
        if not self._sparse_encoder and agent_chunks:
            # Build TF-IDF sparse encoder from all chunks
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer

                corpus = [chunk["text"] for chunk in agent_chunks]
                self._tfidf_vectorizer = TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=1,
                    max_features=10000,  # Limit vocabulary size
                )
                self._tfidf_vectorizer.fit(corpus)

                def _encode(text: str) -> Dict[int, float]:
                    X = self._tfidf_vectorizer.transform([text]) #type: ignore
                    coo = X.tocoo() #type: ignore
                    return {int(j): float(v) for j, v in zip(coo.col, coo.data)}

                self._sparse_encoder = _encode
            except ImportError:
                print("Warning: sklearn not available, sparse vectors won't be generated")
            except Exception as e:
                print(f"Warning: Could not build sparse encoder: {e}")

        # Convert each article chunk to Milvus format
        for article_chunk, agent_chunk in zip(article_chunks, agent_chunks):
            # Build article_id: doc_id:article_number.clause_number
            article_id_parts = [doc_id]
            if article_chunk.article_number is not None:
                article_id_parts.append(str(article_chunk.article_number))
                if article_chunk.clause_number is not None:
                    article_id_parts.append(str(article_chunk.clause_number))
            article_id = ":".join(article_id_parts)

            # Prepare Milvus record
            record: Dict = {
                "article_id": article_id,
                "doc_id": doc_id,
                "doc_type": doc_type,
                "article_number": article_chunk.article_number if article_chunk.article_number is not None else 0,
                "clause_number": str(article_chunk.clause_number) if article_chunk.clause_number is not None else "",
                "text": agent_chunk["text"],
                "title": article_chunk.title or "",
                "year": year or 0,
                "agency": agency or "",
                "status": status,
                "effective_date": effective_date or "",
                "source_url": source_url,
                "last_update": current_time,
                "dense_vector": [],
                "sparse_vector": {},
            }

            # Generate dense embeddings if embedder is provided
            if self._embedder:
                try:
                    embedding = self._embedder(agent_chunk["text"])
                    if embedding and len(embedding) > 0:
                        record["dense_vector"] = embedding
                    else:
                        print(f"Warning: Empty embedding for {article_id}")
                except Exception as e:
                    print(f"Warning: Could not generate embedding for {article_id}: {e}")

            # Generate sparse embeddings if sparse encoder is available
            if self._sparse_encoder:
                try:
                    sparse_vec = self._sparse_encoder(agent_chunk["text"])
                    if sparse_vec and len(sparse_vec) > 0:
                        record["sparse_vector"] = sparse_vec
                except Exception as e:
                    print(f"Warning: Could not generate sparse vector for {article_id}: {e}")

            milvus_data.append(record)

        return milvus_data

