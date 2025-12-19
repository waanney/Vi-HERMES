"""
Ingest ViQuAD CSV data into Milvus

This script:
1. Reads ViQuAD CSV file
2. Uses 'context' column (already chunked) for embedding
3. Parses 'metadata' column for document information
4. Embeds and inserts directly into Milvus (no chunking needed)
"""

from __future__ import annotations

import ast
import csv
import json
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from vihermes.config.settings import get_settings
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
    
    # Try uit_id from row
    if row_id:
        return f"uit_{row_id}"
    
    # Fallback
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


def ingest_viquad_to_milvus(
    csv_path: str | Path,
    batch_size: int = 100,
    max_rows: Optional[int] = None,
    start_row: int = 0,
    recreate_collection: bool = False
):
    """
    Ingest ViQuAD CSV data into Milvus.
    
    Args:
        csv_path: Path to input CSV file
        batch_size: Number of records to insert per batch
        max_rows: Maximum number of rows to process (None for all)
        start_row: Starting row index (for resuming)
    """
    print("=" * 70)
    print("Ingesting ViQuAD CSV to Milvus")
    print("=" * 70)
    print()
    
    # Setup settings
    settings = get_settings()
    
    # Setup Milvus
    print("🔧 Setting up Milvus...")
    milvus_manager = MilvusSchemaManager(
        collection_name=settings.milvus_collection,
        dense_dim=1024,  # multilingual-e5-large dimension
        milvus_uri=f"http://{settings.milvus_host}:{settings.milvus_port}",
    )
    
    if not milvus_manager.connect():
        print("❌ Cannot connect to Milvus. Aborting.")
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
        print("⚠️  Note: If you need to update schema, use --recreate flag")
        # Load collection into memory
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
        return
    
    print()
    
    # Read CSV
    csv_path = Path(csv_path)
    if not csv_path.exists():
        print(f"❌ CSV file not found: {csv_path}")
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
    all_inserted = 0
    successful = 0
    failed = 0
    
    batch_data = []
    
    for idx, row in enumerate(rows, start=start_row + 1):
        try:
            # Get context (already chunked text)
            context = row.get('context', '').strip()
            if not context:
                print(f"[{idx}/{total_rows}] ⚠️  Skipping row {idx}: No context")
                failed += 1
                continue
            
            # Get row ID
            row_id = row.get('id', str(idx))
            
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
            
            # Generate article_id (use row_id as unique identifier)
            # Ensure article_id doesn't exceed 64 chars (max_length for VARCHAR)
            article_id_base = f"{doc_id}:{row_id}"
            article_id = article_id_base[:64] if len(article_id_base) > 64 else article_id_base
            
            # Ensure doc_id doesn't exceed 64 chars
            doc_id = doc_id[:64] if len(doc_id) > 64 else doc_id
            
            # Truncate title and agency to new limits (title: 1000, agency: 500)
            title = title[:1000] if len(title) > 1000 else title
            agency = agency[:500] if len(agency) > 500 else agency
            
            # Embed context
            dense_vector = embedder(context)
            
            # Create sparse vector (empty for now, can be enhanced later)
            sparse_vector = {}
            
            # Build Milvus record
            # Note: INT64 fields cannot be None, use 0 as default
            # Ensure all VARCHAR fields are properly truncated
            record = {
                "article_id": article_id,  # Already truncated to 64 chars
                "doc_id": doc_id,  # Already truncated to 64 chars
                "doc_type": "ViQuAD"[:20],  # Limit to 20 chars (max_length for doc_type)
                "article_number": 0,  # Default to 0 (not applicable for ViQuAD)
                "clause_number": "",  # Empty string (max_length=10)
                "text": context[:65535] if len(context) > 65535 else context,  # Limit to 65535 chars
                "title": (title[:1000] if title else ""),  # Limit to 1000 chars (updated schema)
                "dense_vector": dense_vector,
                "sparse_vector": sparse_vector,
                "year": year if year else 0,  # Default to 0 if no year
                "agency": (agency[:500] if agency else ""),  # Limit to 500 chars (updated schema)
                "status": "effective"[:20],  # Limit to 20 chars
                "effective_date": (effective_date[:20] if effective_date else ""),  # Limit to 20 chars
                "source_url": (source_url[:255] if source_url else ""),  # Limit to 255 chars
                "last_update": 0,  # Default to 0 (can be set to current timestamp if needed)
            }
            
            # Double-check title and agency length (safety check)
            if len(record["title"]) > 1000:
                record["title"] = record["title"][:1000]
            if len(record["agency"]) > 500:
                record["agency"] = record["agency"][:500]
            
            batch_data.append(record)
            
            # Insert batch when full
            if len(batch_data) >= batch_size:
                print(f"[{idx}/{total_rows}] 💾 Inserting batch of {len(batch_data)} records...")
                try:
                    milvus_manager.insert(batch_data)
                    milvus_manager.flush()
                    all_inserted += len(batch_data)
                    successful += len(batch_data)
                    print(f"   ✅ Inserted {len(batch_data)} records")
                except Exception as e:
                    print(f"   ❌ Batch insert error: {e}")
                    failed += len(batch_data)
                batch_data = []
            
            if idx % 100 == 0:
                print(f"[{idx}/{total_rows}] Processed {idx} rows... (✅ {successful}, ❌ {failed})")
        
        except Exception as e:
            print(f"[{idx}/{total_rows}] ❌ Error processing row {idx}: {e}")
            failed += 1
            continue
    
    # Insert remaining batch
    if batch_data:
        print(f"💾 Inserting final batch of {len(batch_data)} records...")
        try:
            milvus_manager.insert(batch_data)
            milvus_manager.flush()
            all_inserted += len(batch_data)
            successful += len(batch_data)
            print(f"   ✅ Inserted {len(batch_data)} records")
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
    print(f"📊 Total records inserted: {all_inserted}")
    print("=" * 70)
    
    # Close connection
    milvus_manager.close()


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Ingest ViQuAD CSV data into Milvus')
    parser.add_argument(
        '--input',
        type=str,
        default='data/viquad_full_levels-1 - Sheet1.csv',
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
    
    ingest_viquad_to_milvus(
        csv_path=args.input,
        batch_size=args.batch_size,
        max_rows=args.max_rows,
        start_row=args.start_row,
        recreate_collection=args.recreate
    )


if __name__ == "__main__":
    main()

