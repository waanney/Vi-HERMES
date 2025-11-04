from __future__ import annotations

from typing import List, Optional

from uraxlaw.lawrag.models import Chunk
from uraxlaw.preprocess.agent_chunker import AgentChunker
from uraxlaw.preprocess.models import DocumentMetadata
from uraxlaw.Agents.relations import extract_all


class DocumentIngestion:
    """
    Handles document ingestion from various sources (PDF, DOCX, HTML, XML).
    Uses AgentChunker for intelligent chunking according to Vietnamese legal structure.
    """

    def __init__(self, chunker: Optional[AgentChunker] = None) -> None:
        """
        Initialize DocumentIngestion.

        Args:
            chunker: Optional AgentChunker instance. If None, creates one with default settings.
        """
        self._chunker = chunker or AgentChunker()

    async def ingest_document(
        self,
        document_id: str,
        text: str,
        metadata: Optional[DocumentMetadata] = None,
    ) -> tuple[List[Chunk], List]:
        """
        Ingest a document and return chunks and extracted relations.

        Args:
            document_id: Unique document identifier
            text: Raw document text
            metadata: Optional document metadata

        Returns:
            Tuple of (chunks, relations)
        """
        # Use AgentChunker for intelligent chunking
        chunks = await self._chunker.chunk(text)

        # Update document_id in chunks
        for chunk in chunks:
            chunk.document_id = document_id

        # Extract relations
        relations = extract_all(source_id=document_id, text=text)

        return chunks, relations

