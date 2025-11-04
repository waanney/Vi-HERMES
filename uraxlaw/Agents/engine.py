from __future__ import annotations

import os
from typing import List

from pydantic import BaseModel

from uraxlaw.Agents.models import AnswerResponse
from uraxlaw.lawrag.models import RetrievalResult
from uraxlaw.Agents.prompt import build_prompt


class LLMClient(BaseModel):
    model: str = "gpt-4o"
    api_key: str | None = None

    def __init__(self, model: str = "gpt-4o", api_key: str | None = None, **kwargs):
        super().__init__(model=model, **kwargs)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

    def complete(self, prompt: str) -> str:
        """Complete prompt using OpenAI API."""
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY env var or pass api_key parameter.")

        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package is required. Install with: pip install openai")

        client = OpenAI(api_key=self.api_key)
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Bạn là chuyên gia pháp lý Việt Nam. Luôn dựa trên ngữ cảnh đã cung cấp, bao gồm cả văn bản luật và dữ liệu đồ thị (các Nút/Quan hệ liên quan). "
                        "Khi trả lời, hãy: (1) chỉ sử dụng thông tin trong ngữ cảnh; (2) trích dẫn chính xác nguồn theo định dạng [Doc: ... | Article/Clause/Point nếu có]; "
                        "(3) nếu lập luận dựa trên quan hệ đồ thị, hãy nêu rõ quan hệ (ví dụ: CITES/REFERENCES/AMENDS) và các node/edge liên quan; (4) không bịa nội dung."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
        )
        return response.choices[0].message.content or ""


class GraphRAGEngine:
    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    def generate(self, query: str, retrieved: List[RetrievalResult]) -> AnswerResponse:
        prompt = build_prompt(query=query, results=retrieved)
        answer_text = self._llm.complete(prompt)

        sources = [
            {"type": "Article", "id": r.chunk.document_id, "chunk": r.chunk.id}
            for r in retrieved
        ]
        trace = ["Hybrid retrieval: vector search + 1-hop graph expansion"]
        return AnswerResponse(answer=answer_text, sources=sources, graph_trace=trace)
