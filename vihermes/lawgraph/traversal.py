from __future__ import annotations

from typing import List, Tuple

from vihermes.lawgraph.models import Edge, Node
from vihermes.lawgraph.neo4j_client import Neo4jClient


class GraphTraversal:
    def __init__(self, client: Neo4jClient) -> None:
        self._client = client

    def neighbors(self, node_id: str, max_hops: int = 1) -> List[Tuple[Node, Edge]]:
        return self._client.expand_related(node_id=node_id, max_hops=max_hops)

