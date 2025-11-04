from __future__ import annotations

from typing import Sequence

from uraxlaw.lawrag.models import RetrievalResult


VI_PROMPT_TEMPLATE = (
    "Bạn là chuyên gia pháp lý Việt Nam. Dưới đây là các điều khoản và văn bản liên quan:\n"
    "{context}\n\n"
    "Câu hỏi: {query}\n\n"
    "Hãy trả lời dựa trên nội dung luật, ghi rõ nguồn trích dẫn (văn bản, điều, khoản, điểm)."
)


def build_context(results: Sequence[RetrievalResult]) -> str:
    lines = []
    for r in results:
        src = r.chunk
        meta = src.document_id
        lines.append(f"[Doc: {meta} | Chunk: {src.id}]\n{src.text}\n")
    return "\n".join(lines)


def build_prompt(query: str, results: Sequence[RetrievalResult]) -> str:
    context = build_context(results)
    return VI_PROMPT_TEMPLATE.format(context=context, query=query)

