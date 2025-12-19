from __future__ import annotations

from typing import Callable, List, Optional

from vihermes.lawgraph.models import Edge, Node
from vihermes.lawrag.models import Chunk
from vihermes.preprocess.models import DocumentMetadata
from vihermes.lawgraph.neo4j_client import Neo4jClient
from vihermes.lawrag.milvus_client import MilvusClient, MilvusSchemaManager
from vihermes.preprocess.ingestion import DocumentIngestion


class PreprocessPipeline:
    """
    Complete preprocessing pipeline for legal documents:
    1. Ingest documents
    2. Segment into chunks
    3. Extract relations
    4. Store in Milvus (vector store)
    5. Store in Neo4j (knowledge graph)
    """

    def __init__(
        self,
        milvus_manager: MilvusSchemaManager,
        neo4j_client: Neo4jClient,
        embedder: Optional[Callable[[str], List[float]]] = None,
    ) -> None:
        self._milvus_manager = milvus_manager
        self._neo4j_client = neo4j_client
        self._ingestion = DocumentIngestion()
        self._embedder = embedder

    async def process_document(
        self,
        document_id: str,
        text: str,
        document_type: str = "Law",
        metadata: Optional[DocumentMetadata] = None,
    ) -> tuple[List[Chunk], List[Edge]]:
        """
        Process a single document through the full pipeline.

        Args:
            document_id: Unique document identifier
            text: Raw document text
            document_type: Type of document (Law, Decree, Circular, etc.)
            metadata: Optional document metadata

        Returns:
            Tuple of (chunks, relations)
        """
        # 1. Ingest and segment using AgentChunker
        chunks, relations = await self._ingestion.ingest_document(
            document_id=document_id, text=text, metadata=metadata
        )

        # 2. Create document node in Neo4j
        props = {}
        if metadata:
            props = {
                k: v
                for k in ["issuing_authority", "effect_date", "field", "status", "source_url"]
                if (v := getattr(metadata, k, None))
            }

        self._neo4j_client.upsert_node(
            label=document_type,
            node_id=document_id,
            properties=props if props else None,
        )

        # 3. Store chunks in Milvus with embeddings
        if self._embedder:
            milvus_data = []
            for chunk in chunks:
                embedding = self._embedder(chunk.text)
                milvus_data.append(
                    {
                        "id": chunk.id,
                        "original_doc_id": chunk.document_id,
                        "text": chunk.text,
                        "source": document_type,
                        "url": metadata.source_url if metadata and metadata.source_url else "",
                        "dense_vec": embedding,
                        "sparse_vec": {},
                    }
                )
            if milvus_data:
                self._milvus_manager.insert(milvus_data)

        # 4. Store relations in Neo4j
        for rel in relations:
            self._neo4j_client.upsert_edge(
                src_label=document_type,
                src_id=rel.source_id,
                relation=rel.relation,
                tgt_label="Law",
                tgt_id=rel.target_id,
            )

        return chunks, relations

    def flush(self) -> None:
        """Flush all pending writes to both stores."""
        self._milvus_manager.flush()

