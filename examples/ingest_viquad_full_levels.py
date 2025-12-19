"""
Ingest ViQuAD Full Levels CSV data into Milvus and Neo4j

This script:
1. Reads ViQuAD CSV file (viquad_full_levels_cleaned.csv)
2. Uses 'context' column for embedding and storing in Milvus
3. Parses 'metadata' column for document information
4. Creates Document nodes in Neo4j from metadata
5. Creates Article nodes for each context and links to Document
"""

from __future__ import annotations

import ast
import csv
import json
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from vihermes.config.settings import get_settings
from vihermes.lawgraph.neo4j_client import Neo4jClient
from vihermes.lawrag.milvus_client import MilvusSchemaManager

load_dotenv()


def parse_metadata(metadata_str: str) -> dict:
    """Parse metadata JSON string to dictionary."""
    if not metadata_str or metadata_str.strip() == "":
        return {}
    
    try:
        # Try JSON first
        return json.loads(metadata_str)
    except json.JSONDecodeError:
        try:
            # Try Python literal eval (for dict-like strings)
            return ast.literal_eval(metadata_str)
        except (ValueError, SyntaxError):
            return {}


def extract_doc_id_from_metadata(metadata: dict, row_id: str) -> str:
    """Extract or generate doc_id from metadata."""
    # Try ky_hieu first
    if metadata.get("ky_hieu"):
        return str(metadata["ky_hieu"])
    
    # Fallback to row_id
    return f"doc_{row_id}"


def extract_year_from_date(date_str: Optional[str]) -> Optional[int]:
    """Extract year from date string (format: DD/MM/YYYY)."""
    if not date_str:
        return None
    
    try:
        # Format: "26/09/2024 09:22" or "26/09/2024"
        parts = date_str.split("/")
        if len(parts) >= 3:
            year = int(parts[2].split()[0])  # Get year part, ignore time if present
            return year
    except (ValueError, IndexError):
        pass
    
    return None


def ingest_viquad_full_levels(
    csv_path: str | Path,
    batch_size: int = 100,
    max_rows: Optional[int] = None,
    start_row: int = 0,
    recreate_collection: bool = False
):
    """
    Ingest ViQuAD Full Levels CSV data into Milvus and Neo4j.
    
    Args:
        csv_path: Path to input CSV file
        batch_size: Number of records to insert per batch
        max_rows: Maximum number of rows to process (None for all)
        start_row: Starting row index (for resuming)
        recreate_collection: Whether to recreate Milvus collection
    """
    print("=" * 70)
    print("Ingesting ViQuAD Full Levels CSV to Milvus and Neo4j")
    print("=" * 70)
    print()
    
    # Setup settings
    settings = get_settings()
    
    # Setup Neo4j
    print("🔧 Setting up Neo4j...")
    neo4j = Neo4jClient(
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password,
    )
    neo4j.init_schema()
    print("✅ Neo4j initialized")
    
    # Setup Milvus
    print("🔧 Setting up Milvus...")
    milvus_manager = MilvusSchemaManager(
        collection_name=settings.milvus_collection,
        dense_dim=1024,  # multilingual-e5-large dimension
        milvus_uri=f"http://{settings.milvus_host}:{settings.milvus_port}",
    )
    
    # Retry connection with exponential backoff
    max_retries = 10
    retry_delay = 5
    
    for attempt in range(max_retries):
        if milvus_manager.connect():
            break
        if attempt < max_retries - 1:
            print(f"⏳ Milvus not ready yet. Waiting {retry_delay} seconds... (attempt {attempt + 1}/{max_retries})")
            time.sleep(retry_delay)
        else:
            print("❌ Cannot connect to Milvus after multiple attempts. Aborting.")
            neo4j.close()
            return
    
    # Check if collection exists, if not create it
    if not milvus_manager.client.has_collection(collection_name=settings.milvus_collection):
        print("📦 Collection does not exist. Creating...")
        milvus_manager.recreate_collection()
    elif recreate_collection:
        print("🔄 Recreating collection with updated schema...")
        milvus_manager.recreate_collection()
    else:
        print("✅ Collection already exists. Using existing collection.")
        try:
            milvus_manager.client.load_collection(collection_name=settings.milvus_collection)
        except Exception as e:
            print(f"⚠️  Warning: Could not load collection: {e}")
    
    print()
    
    # Setup embedder
    print("🔧 Setting up embedder...")
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
        milvus_manager.close()
        neo4j.close()
        return
    
    print()
    
    # Read CSV
    csv_path = Path(csv_path)
    if not csv_path.exists():
        print(f"❌ CSV file not found: {csv_path}")
        milvus_manager.close()
        neo4j.close()
        return
    
    print(f"📖 Reading CSV: {csv_path}")
    rows = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    total_rows = len(rows)
    print(f"📊 Total rows: {total_rows}")
    
    # Limit rows if specified
    if max_rows:
        rows = rows[:max_rows]
        print(f"📝 Processing {len(rows)} rows (limited)")
    else:
        print(f"📝 Processing all {len(rows)} rows")
    
    # Skip to start_row if specified
    if start_row > 0:
        rows = rows[start_row:]
        print(f"⏩ Starting from row {start_row}")
    
    print()
    
    # Process rows in batches
    all_inserted_milvus = 0
    all_inserted_neo4j_docs = 0
    all_inserted_neo4j_articles = 0
    successful = 0
    failed = 0
    
    batch_data = []
    processed_docs = set()  # Track which documents we've already created in Neo4j
    
    for idx, row in enumerate(rows, start=start_row + 1):
        try:
            # Get context (text to embed)
            context = row.get('context', '').strip()
            if not context:
                print(f"[{idx}/{total_rows}] ⚠️  Skipping row {idx}: No context")
                failed += 1
                continue
            
            # Get row ID
            row_id = row.get('id', str(idx))
            uit_id = row.get('uit_id', '')
            
            # Parse metadata
            metadata_str = row.get('metadata', '')
            metadata = parse_metadata(metadata_str)
            
            # Extract information
            doc_id = extract_doc_id_from_metadata(metadata, row_id)
            title = metadata.get('trich_yeu', row.get('title', '')) or ''
            agency = metadata.get('don_vi_ban_hanh', '') or ''
            effective_date = metadata.get('ngay_ban_hanh', '') or ''
            year = extract_year_from_date(effective_date)
            
            # Get source URL
            source_url = (
                metadata.get('Link_Tai') or 
                metadata.get('Link_Goc') or 
                metadata.get('File_Goc') or 
                ''
            )
            
            # Ensure IDs don't exceed limits
            doc_id = doc_id[:64] if len(doc_id) > 64 else doc_id
            article_id_base = f"{doc_id}:{row_id}"
            article_id = article_id_base[:64] if len(article_id_base) > 64 else article_id_base
            
            # Truncate fields to limits
            title = title[:1000] if len(title) > 1000 else title
            agency = agency[:500] if len(agency) > 500 else agency
            
            # Create Document node in Neo4j (only once per doc_id)
            if doc_id not in processed_docs:
                doc_props = {
                    "title": title,
                    "issuing_authority": agency,
                    "effect_date": effective_date[:20] if effective_date else "",
                    "status": "effective",
                    "source_url": source_url[:255] if source_url else "",
                    "year": year if year else 0,
                }
                neo4j.upsert_node(
                    label="Document",
                    node_id=doc_id,
                    properties=doc_props
                )
                processed_docs.add(doc_id)
                all_inserted_neo4j_docs += 1
                
                # Create Agency node if agency exists
                if agency:
                    neo4j.upsert_node(
                        label="Agency",
                        node_id=agency[:64] if len(agency) > 64 else agency,
                        properties={"name": agency}
                    )
                    # Link Document to Agency
                    neo4j.upsert_edge(
                        src_label="Document",
                        src_id=doc_id,
                        relation="ISSUED_BY",
                        tgt_label="Agency",
                        tgt_id=agency[:64] if len(agency) > 64 else agency,
                    )
            
            # Create Article node in Neo4j for this context
            article_props = {
                "text": context[:65535] if len(context) > 65535 else context,
                "title": title,
            }
            neo4j.upsert_node(
                label="Article",
                node_id=article_id,
                properties=article_props
            )
            all_inserted_neo4j_articles += 1
            
            # Link Article to Document
            neo4j.upsert_edge(
                src_label="Document",
                src_id=doc_id,
                relation="HAS_ARTICLE",
                tgt_label="Article",
                tgt_id=article_id,
            )
            
            # Embed context for Milvus
            dense_vector = embedder(context)
            
            # Build Milvus record
            record = {
                "article_id": article_id,
                "doc_id": doc_id,
                "doc_type": "ViQuAD"[:20],
                "article_number": 0,
                "clause_number": "",
                "text": context[:65535] if len(context) > 65535 else context,
                "title": title,
                "dense_vector": dense_vector,
                "sparse_vector": {},
                "year": year if year else 0,
                "agency": agency,
                "status": "effective"[:20],
                "effective_date": effective_date[:20] if effective_date else "",
                "source_url": source_url[:255] if source_url else "",
                "last_update": 0,
            }
            
            batch_data.append(record)
            
            # Insert batch when full
            if len(batch_data) >= batch_size:
                print(f"[{idx}/{total_rows}] 💾 Inserting batch of {len(batch_data)} records...")
                try:
                    milvus_manager.insert(batch_data)
                    milvus_manager.flush()
                    all_inserted_milvus += len(batch_data)
                    successful += len(batch_data)
                    print(f"   ✅ Inserted {len(batch_data)} records to Milvus")
                except Exception as e:
                    print(f"   ❌ Batch insert error: {e}")
                    failed += len(batch_data)
                batch_data = []
            
            if idx % 100 == 0:
                print(f"[{idx}/{total_rows}] Processed {idx} rows... (✅ {successful}, ❌ {failed})")
        
        except Exception as e:
            print(f"[{idx}/{total_rows}] ❌ Error processing row {idx}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
            continue
    
    # Insert remaining batch
    if batch_data:
        print(f"💾 Inserting final batch of {len(batch_data)} records...")
        try:
            milvus_manager.insert(batch_data)
            milvus_manager.flush()
            all_inserted_milvus += len(batch_data)
            successful += len(batch_data)
            print(f"   ✅ Inserted {len(batch_data)} records to Milvus")
        except Exception as e:
            print(f"   ❌ Final batch insert error: {e}")
            failed += len(batch_data)
    
    # Final summary
    print()
    print("=" * 70)
    print("📊 Ingestion Summary")
    print("=" * 70)
    print(f"Total processed: {len(rows)}")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Milvus records inserted: {all_inserted_milvus}")
    print(f"📊 Neo4j Document nodes created: {all_inserted_neo4j_docs}")
    print(f"📊 Neo4j Article nodes created: {all_inserted_neo4j_articles}")
    print("=" * 70)
    
    # Close connections
    milvus_manager.close()
    neo4j.close()


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Ingest ViQuAD Full Levels CSV data into Milvus and Neo4j')
    parser.add_argument(
        '--input',
        type=str,
        default='data/viquad_full_levels_cleaned.csv',
        help='Input CSV file path'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=100,
        help='Number of records to insert per batch'
    )
    parser.add_argument(
        '--max-rows',
        type=int,
        default=None,
        help='Maximum number of rows to process (for testing)'
    )
    parser.add_argument(
        '--start-row',
        type=int,
        default=0,
        help='Starting row index (for resuming)'
    )
    parser.add_argument(
        '--recreate',
        action='store_true',
        help='Recreate collection with updated schema (WARNING: This will delete existing data)'
    )
    
    args = parser.parse_args()
    
    ingest_viquad_full_levels(
        csv_path=args.input,
        batch_size=args.batch_size,
        max_rows=args.max_rows,
        start_row=args.start_row,
        recreate_collection=args.recreate
    )


if __name__ == "__main__":
    main()

