from __future__ import annotations

import json

from vihermes.config.settings import get_settings
from vihermes.Agents.engine import GraphRAGEngine, LLMClient
from vihermes.lawgraph.neo4j_client import Neo4jClient  # type: ignore
from vihermes.lawrag.milvus_client import MilvusClient  # type: ignore
from vihermes.lawrag.hybrid import HybridRetriever


def main() -> None:
    settings = get_settings()

    neo4j = Neo4jClient(
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password,
    )

    # Setup embedder for Milvus (required for search)
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("intfloat/multilingual-e5-large")
        dense_dim = model.get_sentence_embedding_dimension()

        # Create MilvusClient with correct dimension
        milvus = MilvusClient(
            host=settings.milvus_host,
            port=settings.milvus_port,
            collection=settings.milvus_collection,
            dense_dim=dense_dim,  # Use actual embedder dimension
        )

        def embedder(text: str):
            vec = model.encode([text], normalize_embeddings=True)[0]
            return vec.tolist()

        milvus.set_embedder(embedder, auto_detect_dim=True)
        print(f"✅ Embedder configured (dim={dense_dim})")
    except Exception as e:
        print(f"⚠️  Could not setup embedder: {e}")
        print("   Milvus search will return empty results without embedder.")
        # Create MilvusClient with default dimension as fallback
        milvus = MilvusClient(
            host=settings.milvus_host,
            port=settings.milvus_port,
            collection=settings.milvus_collection,
        )

    retriever = HybridRetriever(vector=milvus, graph=neo4j)
    llm = LLMClient(model=settings.llm_model)
    engine = GraphRAGEngine(llm=llm)

    query = "Các nghị định và thông tư hướng dẫn Luật Quản lý thuế 2019?"
    print(f"\n🔍 Query: {query}")
    
    hits = retriever.retrieve(query, k=5)
    print(f"📊 Retrieved {len(hits)} results")
    
    if not hits:
        print("⚠️  No results retrieved. Make sure:")
        print("   1. Milvus has data (run milvus_flow.py to insert sample data)")
        print("   2. Embedder is configured")
        print("   3. Neo4j has nodes (run neo4j_flow.py to create sample nodes)")
    else:
        for i, hit in enumerate(hits, 1):
            print(f"   {i}. {hit.chunk.id} (score: {hit.score:.4f})")
            print(f"      Text: {hit.chunk.text[:80]}...")
    
    answer = engine.generate(query=query, retrieved=hits)
    print(f"\n💬 Answer:\n{answer.answer}\n")
    print(json.dumps(answer.model_dump(exclude_none=True), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()


