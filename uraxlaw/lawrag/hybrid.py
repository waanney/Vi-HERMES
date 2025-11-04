from __future__ import annotations

from typing import List

from uraxlaw.lawrag.models import RetrievalResult
from uraxlaw.lawgraph.neo4j_client import Neo4jClient
from uraxlaw.lawrag.milvus_client import MilvusClient


class HybridRetriever:
    def __init__(self, vector: MilvusClient, graph: Neo4jClient) -> None:
        self._vector = vector
        self._graph = graph

    def retrieve(self, query: str, k: int = 5) -> List[RetrievalResult]:
        vector_hits = self._vector.search(query=query, top_k=k)

        results: List[RetrievalResult] = []
        for hit in vector_hits:
            related = (
                self._graph.expand_related(node_id=hit.chunk.node_id, max_hops=1)
                if hit.chunk.node_id
                else None
            )
            results.append(
                RetrievalResult(
                    chunk=hit.chunk,
                    score=hit.score,
                    related_nodes=[n for n, _ in related] if related else None,
                    related_edges=[e for _, e in related] if related else None,
                )
            )

        return results

