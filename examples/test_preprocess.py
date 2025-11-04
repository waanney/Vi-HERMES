from __future__ import annotations

import asyncio
from pathlib import Path

from uraxlaw.config.settings import get_settings
from uraxlaw.lawgraph.neo4j_client import Neo4jClient
from uraxlaw.lawrag.milvus_client import MilvusSchemaManager
from uraxlaw.preprocess.agent_chunker import AgentChunker
from uraxlaw.preprocess.parser import DocumentParser
from uraxlaw.preprocess.pipeline import PreprocessPipeline


# Sample legal text for testing
SAMPLE_LAW_TEXT = """
QUỐC HỘI
_________________
Luật số: 38/2019/QH14    CỘNG HOÀ XÃ HỘI CHỦ NGHĨA VIỆT NAM
Độc lập - Tự do - Hạnh phúc
___________________________

LUẬT QUẢN LÝ THUẾ

Điều 1. Phạm vi điều chỉnh
Luật này quy định về quản lý thuế; quyền và nghĩa vụ của người nộp thuế, cơ quan quản lý thuế, 
cơ quan nhà nước, tổ chức, cá nhân có liên quan trong quản lý thuế.

Điều 2. Đối tượng áp dụng
1. Luật này áp dụng đối với:
   a) Người nộp thuế;
   b) Cơ quan quản lý thuế;
   c) Cơ quan nhà nước, tổ chức, cá nhân có liên quan trong quản lý thuế.
2. Đối với các khoản thu khác, việc quản lý thực hiện theo quy định của pháp luật về quản lý thuế.

Điều 3. Giải thích từ ngữ
Trong Luật này, các từ ngữ dưới đây được hiểu như sau:
1. "Người nộp thuế" là tổ chức, cá nhân nộp thuế theo quy định của pháp luật về thuế.
2. "Cơ quan quản lý thuế" là cơ quan được giao nhiệm vụ quản lý thuế.
3. "Quản lý thuế" là việc cơ quan quản lý thuế thực hiện các biện pháp quản lý thuế theo quy định của pháp luật.
"""


async def test_agent_chunker():
    """Test AgentChunker với sample text."""
    print("=== Testing AgentChunker ===")
    
    chunker = AgentChunker(model="gpt-4o")
    chunks = await chunker.chunk(SAMPLE_LAW_TEXT)
    
    print(f"\nĐã chunk thành {len(chunks)} chunks:\n")
    for chunk in chunks:
        print(f"- ID: {chunk.id}")
        print(f"  Document ID: {chunk.document_id}")
        print(f"  Node ID: {chunk.node_id}")
        print(f"  Order: {chunk.order}")
        print(f"  Content: {chunk.text[:100]}...")
        print()


async def test_parser():
    """Test DocumentParser với file."""
    print("=== Testing DocumentParser ===")
    
    parser = DocumentParser()
    
    # Test với sample text file
    sample_file = Path("sample_law.txt")
    try:
        with open(sample_file, "w", encoding="utf-8") as f:
            f.write(SAMPLE_LAW_TEXT)
        
        text = parser.parse(sample_file)
        print(f"Đã parse file: {sample_file}")
        print(f"Độ dài text: {len(text)} ký tự")
        print(f"Preview: {text[:200]}...")
        print()
        
        # Cleanup
        sample_file.unlink()
    except Exception as e:
        print(f"Error: {e}")


async def test_full_pipeline():
    """Test full preprocessing pipeline."""
    print("=== Testing Full Preprocessing Pipeline ===")
    
    settings = get_settings()
    
    # Setup Milvus
    milvus_manager = MilvusSchemaManager(
        collection_name=settings.milvus_collection,
        dense_dim=384,  # Match embedding model
        milvus_uri=f"http://{settings.milvus_host}:{settings.milvus_port}",
    )
    
    if not milvus_manager.connect():
        print("Cannot connect to Milvus, skipping pipeline test.")
        return
    
    # Setup Neo4j
    neo4j_client = Neo4jClient(
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password,
    )
    neo4j_client.init_schema()
    
    # Setup embedder (simple hash for demo)
    def simple_embedder(text: str):
        import math
        dense_dim = 384
        vec = [0.0] * dense_dim
        tokens = [t for t in text.lower().split() if t]
        for tok in tokens:
            idx = hash(tok) % dense_dim
            vec[idx] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]
    
    # Create pipeline
    pipeline = PreprocessPipeline(
        milvus_manager=milvus_manager,
        neo4j_client=neo4j_client,
        embedder=simple_embedder,
    )
    
    # Use AgentChunker for chunking
    chunker = AgentChunker(model="gpt-4o")
    chunks = await chunker.chunk(SAMPLE_LAW_TEXT)
    
    print(f"\nĐã chunk thành {len(chunks)} chunks")
    print("\nChunks:")
    for chunk in chunks[:5]:  # Show first 5
        print(f"  - {chunk.id}: {chunk.text[:80]}...")
    
    # Process document (store in Milvus & Neo4j)
    from uraxlaw.preprocess.models import DocumentMetadata
    
    metadata = DocumentMetadata(
        document_id="law_38_2019",
        issuing_authority="Quốc hội",
        effect_date="2020-01-01",
        field="Thuế",
        status="effective",
    )
    
    # Note: PreprocessPipeline expects to do chunking internally,
    # but we can manually store chunks here
    print("\nStoring chunks in Milvus and Neo4j...")
    
    # Store chunks manually
    milvus_data = []
    for chunk in chunks:
        milvus_data.append({
            "id": chunk.id,
            "original_doc_id": chunk.document_id,
            "text": chunk.text,
            "source": "Law",
            "url": "",  # Empty string instead of None for varchar field
            "dense_vec": simple_embedder(chunk.text),
            "sparse_vec": {},
        })
    
    milvus_manager.insert(milvus_data)
    milvus_manager.flush()
    
    # Store document node in Neo4j
    neo4j_client.upsert_node(
        label="Law",
        node_id="law_38_2019",
        properties={
            "issuing_authority": metadata.issuing_authority,
            "effect_date": metadata.effect_date,
            "field": metadata.field,
            "status": metadata.status,
        }
    )
    
    print("✅ Pipeline test completed!")
    print(f"  - Stored {len(chunks)} chunks in Milvus")
    print(f"  - Stored document node in Neo4j")


async def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Preprocess Module")
    print("=" * 60)
    print()
    
    # Test 1: AgentChunker
    try:
        await test_agent_chunker()
    except Exception as e:
        print(f"❌ AgentChunker test failed: {e}")
        print()
    
    # Test 2: DocumentParser
    try:
        await test_parser()
    except Exception as e:
        print(f"❌ DocumentParser test failed: {e}")
        print()
    
    # Test 3: Full Pipeline
    try:
        await test_full_pipeline()
    except Exception as e:
        print(f"❌ Full pipeline test failed: {e}")
        print()


if __name__ == "__main__":
    asyncio.run(main())

