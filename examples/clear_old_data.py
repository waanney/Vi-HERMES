"""
Script to clear old data from Neo4j and Milvus databases.
"""

from __future__ import annotations

from dotenv import load_dotenv

from vihermes.config.settings import get_settings
from vihermes.lawgraph.neo4j_client import Neo4jClient
from vihermes.lawrag.milvus_client import MilvusSchemaManager

load_dotenv()


def clear_neo4j_data():
    """Clear all data from Neo4j."""
    print("=" * 70)
    print("Clearing Neo4j Data")
    print("=" * 70)
    
    settings = get_settings()
    neo4j = Neo4jClient(
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password,
    )
    
    with neo4j._driver.session() as session:
        # Delete all nodes and relationships
        result = session.run("MATCH (n) DETACH DELETE n RETURN count(n) as deleted")
        deleted = result.single()["deleted"]
        print(f"✅ Deleted {deleted} nodes from Neo4j")
    
    print("✅ Neo4j cleared")


def clear_milvus_data():
    """Clear all data from Milvus."""
    print("=" * 70)
    print("Clearing Milvus Data")
    print("=" * 70)
    
    settings = get_settings()
    milvus_manager = MilvusSchemaManager(
        collection_name=settings.milvus_collection,
        dense_dim=1024,
        milvus_uri=f"http://{settings.milvus_host}:{settings.milvus_port}",
    )
    
    # Retry connection with exponential backoff
    import time
    max_retries = 10
    retry_delay = 5  # seconds
    
    for attempt in range(max_retries):
        if milvus_manager.connect():
            break
        if attempt < max_retries - 1:
            print(f"⏳ Milvus not ready yet. Waiting {retry_delay} seconds... (attempt {attempt + 1}/{max_retries})")
            time.sleep(retry_delay)
        else:
            print("❌ Cannot connect to Milvus after multiple attempts. Aborting.")
            return
    
    # Recreate collection (this deletes all data)
    milvus_manager.recreate_collection()
    print("✅ Milvus collection recreated (all data cleared)")
    
    # Close connection
    milvus_manager.close()


def main():
    """Main function to clear all data."""
    print("\n⚠️  WARNING: This will delete ALL data from Neo4j and Milvus!")
    print("Press Ctrl+C to cancel, or wait 3 seconds to continue...\n")
    
    import time
    time.sleep(3)
    
    try:
        clear_neo4j_data()
        print()
        clear_milvus_data()
        print("\n✅ All old data cleared!")
    except Exception as e:
        print(f"\n❌ Error clearing data: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

