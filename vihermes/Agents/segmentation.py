from __future__ import annotations

import re
from typing import Iterable, List

from vihermes.lawrag.models import Chunk


ARTICLE_RE = re.compile(r"(?i)\b(điều)\s+(\d+)")


def segment_to_chunks(document_id: str, text: str) -> List[Chunk]:
    chunks: List[Chunk] = []
    buffer: List[str] = []
    current_article = 0

    def flush() -> None:
        if not buffer:
            return
        content = "\n".join(buffer).strip()
        if not content:
            return
        chunk_id = f"{document_id}_Article_{current_article}_chunk_{len(chunks)}"
        chunks.append(
            Chunk(id=chunk_id, document_id=document_id, node_id=None, text=content)
        )

    for line in _iter_lines(text):
        m = ARTICLE_RE.search(line)
        if m:
            flush()
            buffer = [line]
            current_article = int(m.group(2))
        else:
            buffer.append(line)
    flush()
    return chunks


def _iter_lines(text: str) -> Iterable[str]:
    for ln in text.splitlines():
        yield ln

