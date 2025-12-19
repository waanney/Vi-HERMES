from __future__ import annotations

import os
from typing import List, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from vihermes.lawrag.models import Chunk

load_dotenv()


class ArticleChunk(BaseModel):
    """Chunk following Vietnamese legal document structure: Part -> Chapter -> Section -> Article -> Clause -> Point"""

    article_number: Optional[int] = Field(None, description="Article number (e.g., 1, 2, 3...)")
    clause_number: Optional[int] = Field(None, description="Clause number within the article (e.g., 1, 2, 3...)")
    point_symbol: Optional[str] = Field(None, description="Point symbol (e.g., a, b, c, d...)")
    content: str = Field(
        description="COMPLETE and FULL content of the article/clause/point, must not omit any part. "
        "Must include all sub-points, sub-clauses, and all related details. "
        "MUST NOT only extract the title or first few lines."
    )
    title: Optional[str] = Field(None, description="Title of the article (if available)")


class ChunkedDocument(BaseModel):
    """Document chunked according to Vietnamese legal document standards"""

    chunks: List[ArticleChunk] = Field(description="List of chunks divided by article, clause, and point")


SYSTEM_PROMPT = """
You are an expert in analyzing Vietnamese legal documents.
Your task is to analyze the document and split it into chunks according to the standard structure of Vietnamese legal documents:

Structure: Part -> Chapter -> Section -> Article -> Clause -> Point

IMPORTANT RULES:
1. Each Article is an independent unit with an article number (article_number)
2. Each Article can have multiple Clauses, each clause has a clause number (clause_number)
3. Each Clause can have multiple Points, each point has a symbol (point_symbol) like a, b, c, d...
4. If a paragraph does not have a clear article number, it might be a preamble or enforcement clause, set article_number = None
5. Keep the Vietnamese content as-is, do not translate or modify
6. Each chunk must have clear content, no duplicates

⚠️ IMPORTANT - CONTENT RULES:
1. MUST extract the COMPLETE content of each article/clause/point, do not omit any part
2. MUST NOT only extract the title or first few lines - must extract the FULL content from start to end of each unit
3. If an article has multiple clauses, each clause must be a separate chunk with the COMPLETE content of that clause
4. If a clause has multiple points, each point must be a separate chunk with the COMPLETE content of that point
5. Include all sub-clauses, sub-points, and all related details
6. Ensure each chunk has sufficient context to fully understand the legal meaning
7. Don't create more content, only based on the content of the article/clause/point
Correct Examples:
- "Điều 1. Phạm vi điều chỉnh
Luật này quy định về quản lý thuế; quyền và nghĩa vụ của người nộp thuế, cơ quan quản lý thuế, 
cơ quan nhà nước, tổ chức, cá nhân có liên quan trong quản lý thuế."
  -> article_number=1, title="Phạm vi điều chỉnh", content="Luật này quy định về quản lý thuế; quyền và nghĩa vụ của người nộp thuế, cơ quan quản lý thuế, cơ quan nhà nước, tổ chức, cá nhân có liên quan trong quản lý thuế."

- "Điều 2. Đối tượng áp dụng
1. Luật này áp dụng đối với:
   a) Người nộp thuế;
   b) Cơ quan quản lý thuế;
   c) Cơ quan nhà nước, tổ chức, cá nhân có liên quan trong quản lý thuế.
2. Đối với các khoản thu khác, việc quản lý thực hiện theo quy định của pháp luật về quản lý thuế."
  -> Chunk 1: article_number=2, clause_number=1, content="Luật này áp dụng đối với: a) Người nộp thuế; b) Cơ quan quản lý thuế; c) Cơ quan nhà nước, tổ chức, cá nhân có liên quan trong quản lý thuế."
  -> Chunk 2: article_number=2, clause_number=2, content="Đối với các khoản thu khác, việc quản lý thực hiện theo quy định của pháp luật về quản lý thuế."

WRONG Examples (MUST NOT do):
- Only extracting: "Phạm vi điều chỉnh\nLuật này quy định về quản lý thuế;" (missing the rest)
- Only extracting: "Đối tượng áp dụng\n1. Luật này áp dụng đối với:" (missing points a, b, c)
"""


class AgentChunker:
    """
    Agent that uses LLM to chunk legal documents according to Vietnamese legal standards.
    """

    def __init__(self, model: str = "gpt-4o", api_key: Optional[str] = None) -> None:
        """
        Initialize AgentChunker.

        Args:
            model: LLM model name (default: gpt-4o)
            api_key: OpenAI API key (if None, uses OPENAI_API_KEY env var)
        """
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY env var or pass api_key parameter.")

        llm_model = OpenAIChatModel(model_name=model, provider=OpenAIProvider(api_key=api_key))
        self._agent = Agent(
            model=llm_model,
            output_type=ChunkedDocument,
            system_prompt=SYSTEM_PROMPT,
        )

    async def chunk(self, text: str) -> List[Chunk]:
        """
        Chunk legal document according to Vietnamese legal standards.

        Args:
            text: Legal document text to be chunked

        Returns:
            List of Chunk objects
        """
        result = await self._agent.run(text)
        chunked_doc = result.output

        chunks: List[Chunk] = []
        for idx, article_chunk in enumerate(chunked_doc.chunks):
            # Build chunk_id: Article_X_Clause_Y_Point_Z
            parts = []
            if article_chunk.article_number is not None:
                parts.append(f"Article_{article_chunk.article_number}")
            if article_chunk.clause_number is not None:
                parts.append(f"Clause_{article_chunk.clause_number}")
            if article_chunk.point_symbol:
                parts.append(f"Point_{article_chunk.point_symbol}")
            chunk_id = "_".join(parts) if parts else f"chunk_{idx}"

            # Build document_id and node_id
            document_id = (
                f"law_article_{article_chunk.article_number}"
                if article_chunk.article_number
                else "law_general"
            )
            node_id = "_".join(parts) if parts else None

            # Build content with title if available
            content = f"{article_chunk.title}\n{article_chunk.content}" if article_chunk.title else article_chunk.content

            chunks.append(
                Chunk(
                    id=chunk_id,
                    document_id=document_id,
                    node_id=node_id,
                    text=content,
                    order=idx,
                )
            )

        return chunks

