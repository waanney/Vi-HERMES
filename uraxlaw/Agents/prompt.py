from __future__ import annotations

from typing import Sequence

from uraxlaw.lawrag.models import RetrievalResult


VI_PROMPT_TEMPLATE = (
    "Bạn là chuyên gia pháp lý Việt Nam. Dưới đây là ngữ cảnh gồm văn bản và dữ liệu đồ thị (các Nút/Quan hệ) liên quan:\n"
    "{context}\n\n"
    "Câu hỏi: {query}\n\n"
    "Yêu cầu trả lời: sử dụng cả nội dung văn bản và dữ liệu đồ thị khi lập luận; trích dẫn rõ ràng nguồn (văn bản, điều, khoản, điểm) và nêu quan hệ đồ thị liên quan nếu có (ví dụ: CITES/REFERENCES/AMENDS). Không bịa."
)


def build_context(results: Sequence[RetrievalResult]) -> str:
    lines = []
    for r in results:
        src = r.chunk
        meta = src.document_id
        # Văn bản nguồn
        lines.append(f"[Doc: {meta} | Chunk: {src.id}]\n{src.text}")
        # Dữ liệu đồ thị liên quan (nếu có)
        if r.related_nodes:
            node_lines = []
            for n in r.related_nodes:
                title = f" | title: {n.title}" if getattr(n, "title", None) else ""
                node_lines.append(f"- Node: {n.id} | type: {n.type}{title}")
            lines.append("[Graph Nodes]\n" + "\n".join(node_lines))
        if r.related_edges:
            edge_lines = []
            for e in r.related_edges:
                edge_lines.append(f"- Edge: {e.source_id} -{e.relation}-> {e.target_id}")
            lines.append("[Graph Edges]\n" + "\n".join(edge_lines))
        lines.append("")  # blank line separator
    return "\n".join(lines)


def build_prompt(query: str, results: Sequence[RetrievalResult]) -> str:
    context = build_context(results)
    return VI_PROMPT_TEMPLATE.format(context=context, query=query)
