from __future__ import annotations

import re
from typing import List

from vihermes.lawgraph.models import Edge


REF_RE = re.compile(r"(?i)(Nghị định|Luật|Thông tư)\s+([0-9]{1,4}/[0-9]{4})")


def extract_references(source_id: str, text: str) -> List[Edge]:
    edges: List[Edge] = []
    for m in REF_RE.finditer(text):
        kind = m.group(1)
        code = m.group(2)
        target_id = f"{kind}_{code}"
        edges.append(Edge(source_id=source_id, target_id=target_id, relation="REFERENCES"))
    return edges


def extract_all(source_id: str, text: str) -> List[Edge]:
    return extract_references(source_id, text)

