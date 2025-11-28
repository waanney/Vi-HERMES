"""
End-to-end test flow: Parser -> Milvus Ingest

This script demonstrates the complete flow:
1. Parse document with metadata extraction
2. Chunk document for Milvus (with embeddings)
3. Insert chunks into Milvus
4. Query and verify data
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from dotenv import load_dotenv

from uraxlaw.config.settings import get_settings
from uraxlaw.lawrag.milvus_client import MilvusSchemaManager
from uraxlaw.preprocess.milvus_chunker import MilvusChunker
from uraxlaw.preprocess.parser import DocumentParser
from uraxlaw.preprocess.models import DocumentMetadata

load_dotenv()


async def test_parser_to_milvus_flow():
    """Test complete flow from parser to Milvus ingestion."""
    print("=" * 70)
    print("End-to-End Test: Parser -> Milvus Ingest")
    print("=" * 70)
    print()

    settings = get_settings()

    # Step 1: Setup Milvus
    print("🔧 Step 1: Setting up Milvus...")
    milvus_manager = MilvusSchemaManager(
        collection_name=settings.milvus_collection,
        dense_dim=1024,  # multilingual-e5-large dimension
        milvus_uri=f"http://{settings.milvus_host}:{settings.milvus_port}",
    )
    if not milvus_manager.connect():
        print("❌ Cannot connect to Milvus. Aborting.")
        return
    milvus_manager.recreate_collection()
    print("✅ Milvus initialized\n")

    # Step 2: Setup embedder
    print("🔧 Step 2: Setting up embedder...")
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

    # Step 3: Setup parser and chunker
    print("🔧 Step 3: Setting up parser and chunker...")
    parser = DocumentParser()
    # MilvusChunker will build sparse encoder automatically from chunks
    milvus_chunker = MilvusChunker(embedder=embedder)
    print("✅ Parser and chunker initialized (sparse encoder will be built from chunks)\n")

    # Get project root
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data" / "data_txt_clean 2" / "data_txt_clean"

    # Get all .txt files from the new data directory
    print(f"📁 Scanning data directory: {data_dir}")
    all_txt_files = sorted(list(data_dir.glob("*.txt")))  # Sort for consistent processing
    print(f"📊 Found {len(all_txt_files)} .txt files")
    
    # Process ALL files
    test_files = []
    for file_path in all_txt_files:
        test_files.append({
            "file": file_path.name,
            "file_path": file_path,  # Full path
            "doc_id": None,  # Will be extracted from metadata
            "doc_type": None,  # Will be extracted from metadata
            "metadata": None,  # Will be extracted from file
        })
    
    print(f"📝 Processing ALL {len(test_files)} files\n")

    all_inserted_records = 0
    successful_files = 0
    failed_files = 0

    total_files = len(test_files)
    print(f"🚀 Starting to process {total_files} files...\n")

    for idx, test_case in enumerate(test_files, 1):
        filename = test_case["file"]
        # Use full path if provided, otherwise construct from data_dir
        file_path = test_case.get("file_path") or (data_dir / filename)
        default_doc_id = test_case.get("doc_id")
        default_doc_type = test_case.get("doc_type")
        provided_metadata = test_case.get("metadata")

        if not file_path.exists():
            print(f"❌ File not found: {file_path}")
            continue

        print(f"\n{'─' * 70}")
        print(f"📄 [{idx}/{total_files}] Processing: {filename}")
        print(f"📁 File path: {file_path}")
        print(f"{'─' * 70}\n")

        try:
            # Step 4: Parse document with metadata extraction
            print("📖 Step 4: Parsing document with metadata extraction...")
            text, extracted_metadata = parser.parse_with_metadata(file_path)
            print(f"✅ Parsed {len(text)} characters")

            # Use extracted metadata as primary, fallback to provided metadata
            if extracted_metadata:
                print("✅ Metadata extracted from file:")
                print(f"   - document_id: {extracted_metadata.document_id}")
                print(f"   - issuing_authority: {extracted_metadata.issuing_authority}")
                print(f"   - effect_date: {extracted_metadata.effect_date}")
                print(f"   - field: {extracted_metadata.field}")
                print(f"   - status: {extracted_metadata.status}")

                # Use extracted metadata as primary
                metadata = extracted_metadata
                doc_id = extracted_metadata.document_id

                # Infer doc_type from doc_id if not set
                if not doc_id or doc_id == "UNKNOWN":
                    doc_id = default_doc_id
                    metadata.document_id = default_doc_id

                # Infer doc_type from doc_id
                doc_type = parser._infer_doc_type(doc_id) if doc_id else default_doc_type

                # Fill missing fields from provided metadata (if available)
                if provided_metadata:
                    if not metadata.issuing_authority and provided_metadata.issuing_authority:
                        metadata.issuing_authority = provided_metadata.issuing_authority
                    if not metadata.effect_date and provided_metadata.effect_date:
                        metadata.effect_date = provided_metadata.effect_date
                    if not metadata.field and provided_metadata.field:
                        metadata.field = provided_metadata.field
                    if not metadata.status and provided_metadata.status:
                        metadata.status = provided_metadata.status
            else:
                print("⚠️  No metadata found in file")
                if provided_metadata:
                    print("   Using provided metadata")
                    metadata = provided_metadata
                else:
                    print("   Creating default metadata")
                    from uraxlaw.preprocess.models import DocumentMetadata
                    metadata = DocumentMetadata(
                        document_id=default_doc_id,
                        issuing_authority=None,
                        effect_date=None,
                        field=None,
                        status="effective",
                    )
                doc_id = default_doc_id
                doc_type = default_doc_type

            print()

            # Step 5: Chunk document for Milvus
            print("🔗 Step 5: Chunking document for Milvus...")
            try:
                # Add timeout for chunking (5 minutes per file)
                milvus_data = await asyncio.wait_for(
                    milvus_chunker.chunk_for_milvus(
                        text=text,
                        doc_id=doc_id,
                        doc_type=doc_type,
                        metadata=metadata,
                    ),
                    timeout=300.0  # 5 minutes timeout
                )
                print(f"✅ Generated {len(milvus_data)} Milvus records")
            except asyncio.TimeoutError:
                print(f"⏱️  Timeout: Chunking took too long for {filename}, skipping...")
                failed_files += 1
                continue
            except Exception as e:
                print(f"❌ Chunking error for {filename}: {e}")
                failed_files += 1
                continue

            # Display statistics
            records_with_dense = sum(1 for r in milvus_data if r.get("dense_vector") and len(r.get("dense_vector", [])) > 0)
            records_with_sparse = sum(1 for r in milvus_data if r.get("sparse_vector") and len(r.get("sparse_vector", {})) > 0)
            print(f"   - Records with dense vectors: {records_with_dense}/{len(milvus_data)}")
            print(f"   - Records with sparse vectors: {records_with_sparse}/{len(milvus_data)}")

            # Display sample records
            print("\n📊 Sample records:")
            for i, record in enumerate(milvus_data[:3], 1):
                print(f"   Record {i}:")
                print(f"     - article_id: {record['article_id']}")
                print(f"     - doc_id: {record['doc_id']}")
                print(f"     - article_number: {record['article_number']}")
                print(f"     - clause_number: {record['clause_number']}")
                print(f"     - title: {record['title'][:50]}..." if record['title'] else "     - title: (empty)")
                print(f"     - text: {record['text'][:100]}...")
                print(f"     - dense_vector: {len(record.get('dense_vector', []))} dims")
                sparse_vec = record.get('sparse_vector', {})
                if isinstance(sparse_vec, dict):
                    print(f"     - sparse_vector: {len(sparse_vec)} non-zero entries")
                else:
                    print(f"     - sparse_vector: {type(sparse_vec)}")

            if len(milvus_data) > 3:
                print(f"   ... and {len(milvus_data) - 3} more records")

            print()

            # Step 6: Insert into Milvus
            print("💾 Step 6: Inserting into Milvus...")
            milvus_manager.insert(milvus_data)
            milvus_manager.flush()
            print(f"✅ Inserted {len(milvus_data)} records into Milvus")
            all_inserted_records += len(milvus_data)
            successful_files += 1

            # Step 7: Query and verify
            print("🔍 Step 7: Querying and verifying data...")

            # Load collection into memory
            if milvus_manager.client:
                try:
                    milvus_manager.client.load_collection(collection_name=settings.milvus_collection)
                except Exception as e:
                    print(f"⚠️  Collection load warning: {e}")

            # Query by doc_id
            try:
                if not milvus_manager.client:
                    print("⚠️  Milvus client not available")
                else:
                    query_result = milvus_manager.client.query(
                        collection_name=settings.milvus_collection,
                        filter=f'doc_id == "{doc_id}"',
                        limit=5,
                        output_fields=["article_id", "doc_id", "article_number", "title", "text"],
                    )

                    if query_result:
                        print(f"✅ Found {len(query_result)} records for doc_id: {doc_id}")
                        print("\n📋 Sample query results:")
                        for i, record in enumerate(query_result[:3], 1):
                            print(f"   Result {i}:")
                            print(f"     - article_id: {record.get('article_id')}")
                            print(f"     - article_number: {record.get('article_number')}")
                            print(f"     - title: {record.get('title', '')[:50]}..." if record.get('title') else "     - title: (empty)")
                    else:
                        print(f"⚠️  No records found for doc_id: {doc_id}")
            except Exception as e:
                print(f"⚠️  Query error: {e}")

            # Test search
            if milvus_data and len(milvus_data) > 0 and milvus_manager.client:
                try:
                    # Use first record's text for search test
                    test_text = milvus_data[0]["text"][:200]  # First 200 chars
                    test_embedding = embedder(test_text)
                    
                    # Generate sparse vector (simple TF-IDF-like approach)
                    from sklearn.feature_extraction.text import TfidfVectorizer
                    import numpy as np
                    
                    # Build sparse vector for query
                    # For simplicity, use a basic approach
                    sparse_query = {}  # Would need proper sparse vector generation
                    
                    # Use MilvusSchemaManager's search method with proper parameters
                    search_results = milvus_manager.search(
                        dense_query_vector=test_embedding,
                        sparse_query_vector=sparse_query,
                        limit=5,
                    )

                    if search_results and len(search_results) > 0:
                        print(f"\n✅ Search test successful (found {len(search_results)} results)")
                        print("📋 Top search results:")
                        for i, result in enumerate(search_results[:3], 1):
                            # Search results from MilvusSchemaManager.search are dicts
                            if isinstance(result, dict):
                                article_id = result.get('article_id', 'N/A')
                                score = result.get('score', 0.0)
                                title = result.get('title', '')
                                print(f"   {i}. article_id: {article_id}, score: {score:.4f}")
                                print(f"      title: {title[:50]}..." if title else "      title: (empty)")
                            else:
                                print(f"   {i}. Result: {result}")
                    else:
                        print("⚠️  Search test returned no results")
                except Exception as e:
                    print(f"⚠️  Search error: {e}")
                    import traceback
                    traceback.print_exc()

            print()

        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")
            failed_files += 1
            import traceback

            traceback.print_exc()
            print()

    # Final summary
    print(f"\n{'=' * 70}")
    print("📊 Final Summary")
    print(f"{'=' * 70}")
    print(f"✅ Successfully processed: {successful_files}/{total_files} files")
    print(f"❌ Failed: {failed_files}/{total_files} files")
    print(f"📊 Total records inserted: {all_inserted_records}")
    print()

    # Database statistics
    print("🔍 Database Statistics:")
    try:
        if not milvus_manager.client:
            print("⚠️  Milvus client not available")
        else:
            # Count by doc_id - query all records
            all_docs = milvus_manager.client.query(
                collection_name=settings.milvus_collection,
                filter="",
                output_fields=["doc_id"],
                limit=10000,  # Adjust limit as needed
            )
            
            if all_docs:
                total_count = len(all_docs)
                print(f"   - Total records queried: {total_count}")
                
                doc_counts = {}
                for doc in all_docs:
                    doc_id = doc.get("doc_id")
                    if doc_id:
                        doc_counts[doc_id] = doc_counts.get(doc_id, 0) + 1

                print(f"\n   Records by document:")
                for doc_id, count in sorted(doc_counts.items()):
                    print(f"     - {doc_id}: {count} records")
            else:
                print("   - No records found in collection")

    except Exception as e:
        print(f"⚠️  Could not query database statistics: {e}")

    print(f"\n{'=' * 70}")
    print("✅ End-to-end test completed!")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    asyncio.run(test_parser_to_milvus_flow())

