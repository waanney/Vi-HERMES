from __future__ import annotations

import asyncio
import os
from pathlib import Path

# --- THÊM ĐOẠN NÀY ĐỂ NẠP .ENV ---
from dotenv import load_dotenv
load_dotenv()  # Nạp biến môi trường từ file .env ngay lập tức
# ---------------------------------

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
2. "Mã số thuế" là một dãy số, chữ cái hoặc ký tự khác do cơ quan quản lý thuế cấp cho người nộp thuế 
để quản lý thuế.
"""


async def test_agent_chunker():
    """Test AgentChunker class."""
    print("-" * 50)
    print("Testing: AgentChunker")
    print("-" * 50)

    # Initialize chunker
    try:
        chunker = AgentChunker()
        print("✅ AgentChunker initialized")
    except Exception as e:
        print(f"❌ Failed to initialize AgentChunker: {e}")
        return

    # Test chunking
    print("Running semantic chunking...")
    try:
        chunks = await chunker.chunk(SAMPLE_LAW_TEXT)
        print(f"✅ Generated {len(chunks)} chunks")
        
        for i, chunk in enumerate(chunks):
            print(f"\nChunk {i+1}:")
            print(f"  - Text preview: {chunk.text[:100]}...")
            print(f"  - Title: {chunk.metadata.get('title', 'N/A')}")
            print(f"  - Keywords: {chunk.metadata.get('keywords', [])}")
            
    except Exception as e:
        print(f"❌ Chunking failed: {e}")


async def test_parser():
    """Test DocumentParser with pipeline integration."""
    print("\n" + "-" * 50)
    print("Testing: DocumentParser & Pipeline")
    print("-" * 50)

    settings = get_settings()
    
    # 1. Setup Neo4j & Milvus Clients
    neo4j_client = Neo4jClient(
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password,
    )
    
    # Connect Milvus
    milvus_manager = MilvusSchemaManager(
        collection_name=settings.milvus_collection,
        dense_dim=1024, 
        milvus_uri=f"http://{settings.milvus_host}:{settings.milvus_port}",
    )
    
    # Mock embedding function for testing (returns random vector)
    def simple_embedder(text: str):
        import random
        return [random.random() for _ in range(1024)]
    
    milvus_manager.connect()
    
    # 2. Run Parser
    parser = DocumentParser()
    
    # Create a temporary file for testing
    test_file = Path("test_law.txt")
    test_file.write_text(SAMPLE_LAW_TEXT, encoding="utf-8")
    
    print(f"Parsing file: {test_file}")
    metadata = parser.parse(test_file)
    print(f"✅ Parsed metadata: {metadata.title}")
    
    # Clean up
    if test_file.exists():
        test_file.unlink()

    # 3. Integrate with Chunker (Simulation)
    chunker = AgentChunker()
    chunks = await chunker.chunk(SAMPLE_LAW_TEXT)
    
    # 4. Store in DBs (Simulation)
    print(f"Storing {len(chunks)} chunks to databases...")
    
    # Milvus insert
    milvus_data = []
    for i, chunk in enumerate(chunks):
        milvus_data.append({
            "id": f"test_chunk_{i}",
            "doc_id": metadata.document_id,
            "text": chunk.text,
            "source": "Law",
            "url": "",  
            "dense_vec": simple_embedder(chunk.text),
            "sparse_vec": {},
        })
    
    # Lưu ý: Đoạn này giả lập insert, nếu Milvus chưa setup schema có thể lỗi
    # milvus_manager.insert(milvus_data) 
    print("✅ [Mock] Milvus insert called")
    
    # Neo4j insert
    # neo4j_client.upsert_node(...)
    print("✅ [Mock] Neo4j insert called")
    
    print("✅ Pipeline test completed successfully!")


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
        import traceback
        traceback.print_exc()
        print()
    
    # Test 2: DocumentParser
    try:
        await test_parser()
    except Exception as e:
        print(f"❌ DocumentParser test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
