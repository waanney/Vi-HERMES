from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from vihermes.Agents.engine import GraphRAGEngine, LLMClient
from vihermes.config.settings import get_settings
from vihermes.lawgraph.neo4j_client import Neo4jClient
from vihermes.lawrag.hybrid import HybridRetriever
from vihermes.lawrag.milvus_client import MilvusClient, MilvusSchemaManager
from vihermes.preprocess.agent_chunker import AgentChunker
from vihermes.preprocess.models import DocumentMetadata
from vihermes.preprocess.parser import DocumentParser
from vihermes.preprocess.pipeline import PreprocessPipeline

load_dotenv()


async def test_parse_store_and_query():
    """Test parsing, storing vào database và query."""
    print("=" * 70)
    print("Testing: Parse → Store → Query")
    print("=" * 70)
    print()

    settings = get_settings()

    # Setup databases
    print("🔧 Setting up databases...")
    
    # Neo4j
    neo4j = Neo4jClient(
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password,
    )
    neo4j.init_schema()
    print("✅ Neo4j initialized")

    # Milvus
    milvus_manager = MilvusSchemaManager(
        collection_name=settings.milvus_collection,
        dense_dim=1024,  # multilingual-e5-large dimension
        milvus_uri=f"http://{settings.milvus_host}:{settings.milvus_port}",
    )
    if not milvus_manager.connect():
        print("❌ Cannot connect to Milvus. Aborting.")
        return
    milvus_manager.recreate_collection()
    print("✅ Milvus initialized")

    # Setup embedder
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("intfloat/multilingual-e5-large")
        dense_dim = model.get_sentence_embedding_dimension()

        def embedder(text: str):
            vec = model.encode([text], normalize_embeddings=True)[0]
            return vec.tolist()

        print(f"✅ Embedder configured (dim={dense_dim})")
    except Exception as e:
        print(f"❌ Could not setup embedder: {e}")
        return

    # Setup pipeline
    pipeline = PreprocessPipeline(
        milvus_manager=milvus_manager,
        neo4j_client=neo4j,
        embedder=embedder,
    )

    # Parse và store 2 files
    parser = DocumentParser()
    chunker = AgentChunker(model="gpt-4o")

    files = [
        ("../data/sample_law_1.txt", "law_38_2019", "Law", "Quốc hội", "2020-01-01"),
        ("../data/sample_law_2.txt", "decree_126_2020", "Decree", "Chính phủ", "2020-07-01"),
    ]

    all_chunks = []
    for file_path, doc_id, doc_type, authority, effect_date in files:
        print(f"\n{'='*70}")
        print(f"📄 Processing: {file_path} - {doc_id}")
        print(f"{'='*70}")

        try:
            # 1. Parse
            print(f"\n1️⃣  Parsing file...")
            text = parser.parse(file_path)
            print(f"   ✅ Parsed {len(text)} characters")

            # 2. Chunking với Agent
            print(f"\n2️⃣  Chunking với AgentChunker...")
            chunks = await chunker.chunk(text)
            print(f"   ✅ Chunked thành {len(chunks)} chunks")

            # Update document_id trong chunks
            for chunk in chunks:
                chunk.document_id = doc_id
            all_chunks.extend(chunks)

            # 3. Create metadata
            metadata = DocumentMetadata(
                document_id=doc_id,
                issuing_authority=authority,
                effect_date=effect_date,
                field="Thuế",
                status="effective",
            )

            # 4. Store vào databases
            print(f"\n3️⃣  Storing vào Milvus & Neo4j...")
            
            # Store chunks vào Milvus với embeddings
            milvus_data = []
            for chunk in chunks:
                embedding = embedder(chunk.text)
                milvus_data.append({
                    "id": chunk.id,
                    "original_doc_id": chunk.document_id,
                    "text": chunk.text,
                    "source": doc_type,
                    "url": "",  # Empty string instead of None for varchar field
                    "dense_vec": embedding,
                    "sparse_vec": {},
                })
            if milvus_data:
                milvus_manager.insert(milvus_data)
                milvus_manager.flush()
                print(f"   ✅ Stored {len(milvus_data)} chunks vào Milvus")

            # Store document node vào Neo4j
            props = {
                "issuing_authority": metadata.issuing_authority,
                "effect_date": metadata.effect_date,
                "field": metadata.field,
                "status": metadata.status,
            }
            neo4j.upsert_node(
                label=doc_type,
                node_id=doc_id,
                properties=props,
            )
            print(f"   ✅ Stored document node vào Neo4j")

            # Extract và store relations
            from vihermes.Agents.relations import extract_all
            relations = extract_all(source_id=doc_id, text=text)
            for rel in relations:
                neo4j.upsert_edge(
                    src_label=doc_type,
                    src_id=rel.source_id,
                    relation=rel.relation,
                    tgt_label="Law",  # Default target type
                    tgt_id=rel.target_id,
                )
            print(f"   ✅ Stored {len(relations)} relations vào Neo4j")

            # 5. Create relation từ Decree đến Law
            if doc_type == "Decree":
                neo4j.upsert_edge(
                    src_label="Decree",
                    src_id="decree_126_2020",
                    relation="GUIDES",
                    tgt_label="Law",
                    tgt_id="law_38_2019",
                )
                print(f"   ✅ Created GUIDES relation: Decree → Law")

        except Exception as e:
            print(f"   ❌ Error: {e}")
            import traceback
            traceback.print_exc()

    # Flush to make data searchable
    print(f"\n{'='*70}")
    print("💾 Flushing data to make searchable...")
    milvus_manager.flush()
    # Ensure collection is loaded into memory
    if milvus_manager.client:
        try:
            milvus_manager.client.load_collection(collection_name=settings.milvus_collection)
            print("✅ Collection loaded into memory")
            # Check collection stats
            try:
                stats = milvus_manager.client.get_collection_stats(collection_name=settings.milvus_collection)
                print(f"📊 Collection stats: {stats}")
            except Exception as e:
                print(f"⚠️  Could not get collection stats: {e}")
        except Exception as e:
            print(f"⚠️  Collection load warning: {e}")
    print("✅ Data flushed and ready for search")

    # Test query
    print(f"\n{'='*70}")
    print("🔍 Testing Query")
    print(f"{'='*70}")

    # Setup query components
    milvus_client = MilvusClient(
        host=settings.milvus_host,
        port=settings.milvus_port,
        collection=settings.milvus_collection,
        dense_dim=dense_dim,
    )
    milvus_client.set_embedder(embedder, auto_detect_dim=True)

    retriever = HybridRetriever(vector=milvus_client, graph=neo4j)
    llm = LLMClient(model=settings.llm_model)
    engine = GraphRAGEngine(llm=llm)

    # Test queries - dựa trên nội dung Luật Quản lý thuế 38/2019/QH14
    queries = [
        # Câu hỏi về định nghĩa (Điều 3)

        "Cơ quan quản lý thuế là gì?",
        "Quản lý thuế được hiểu như thế nào?",
        
        # Câu hỏi về đối tượng áp dụng (Điều 2)
        "Luật Quản lý thuế 38/2019 áp dụng đối với những đối tượng nào?",
        
        # Câu hỏi về người nộp thuế (Điều 5)
        "Người nộp thuế bao gồm những ai?",
        
        # Câu hỏi về quyền (Điều 6)
        "Người nộp thuế có những quyền gì?",
        
        # Câu hỏi về nguyên tắc (Điều 4)
        "Nguyên tắc quản lý thuế là gì?",
        
        # Câu hỏi về phạm vi (Điều 1)
        "Luật Quản lý thuế 38/2019 quy định về những vấn đề gì?",
        
        # Câu hỏi về Nghị định 126/2020/NĐ-CP
        "Kê khai thuế là gì theo Nghị định 126/2020?",
        "Nộp thuế được hiểu như thế nào?",
        "Ấn định thuế là gì?",
        "Người nộp thuế phải đăng ký thuế trong thời hạn bao lâu?",
        "Hồ sơ đăng ký thuế bao gồm những gì?",
        "Người nộp thuế có thể kê khai thuế bằng những cách nào?",
        "Thời hạn nộp thuế được quy định như thế nào?",
        "Người nộp thuế có thể nộp thuế bằng những phương thức nào?",
    ]

    for query in queries:
        print(f"\n{'─'*70}")
        print(f"❓ Query: {query}")
        print(f"{'─'*70}")

        try:
            # Retrieve
            hits = retriever.retrieve(query, k=3)
            print(f"📊 Retrieved {len(hits)} results")

            if hits:
                for i, hit in enumerate(hits, 1):
                    print(f"   {i}. {hit.chunk.id} (score: {hit.score:.4f})")
                    print(f"      Text: {hit.chunk.text[:100]}...")

                # Generate answer
                print(f"\n💬 Generating answer...")
                answer = engine.generate(query=query, retrieved=hits)
                print(f"✅ Answer:\n{answer.answer}")
                print(f"\n📚 Sources:")
                for src in answer.sources:
                    print(f"   - {src}")
                if answer.graph_trace:
                    print(f"\n🔗 Graph trace:")
                    for trace in answer.graph_trace:
                        print(f"   - {trace}")
            else:
                print("⚠️  No results found")

        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*70}")
    print("✅ Test completed!")
    print(f"{'='*70}")


if __name__ == "__main__":
    asyncio.run(test_parse_store_and_query())

