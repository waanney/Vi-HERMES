"""
End-to-end test flow: Parser -> Neo4j Ingest

This script demonstrates the complete flow:
1. Parse document with metadata extraction
2. Chunk document for Neo4j
3. Insert nodes and edges into Neo4j
4. Query and verify data
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from dotenv import load_dotenv

from uraxlaw.config.settings import get_settings
from uraxlaw.lawgraph.neo4j_client import Neo4jClient
from uraxlaw.preprocess.neo4j_chunker import Neo4jChunker
from uraxlaw.preprocess.parser import DocumentParser
from uraxlaw.preprocess.models import DocumentMetadata

load_dotenv()


async def test_parser_to_neo4j_flow():
    """Test complete flow from parser to Neo4j ingestion."""
    print("=" * 70)
    print("End-to-End Test: Parser -> Neo4j Ingest")
    print("=" * 70)
    print()

    settings = get_settings()

    # Step 1: Setup Neo4j
    print("🔧 Step 1: Setting up Neo4j...")
    neo4j = Neo4jClient(
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password,
    )
    neo4j.init_schema()
    print("✅ Neo4j initialized\n")

    # Step 2: Setup parser and chunker
    print("🔧 Step 2: Setting up parser and chunker...")
    parser = DocumentParser()
    neo4j_chunker = Neo4jChunker()
    print("✅ Parser and chunker initialized\n")

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

    all_inserted_nodes = 0
    all_inserted_edges = 0
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
            # Step 3: Parse document with metadata extraction
            print("📖 Step 3: Parsing document with metadata extraction...")
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

            # Step 4: Chunk document for Neo4j
            print("🔗 Step 4: Chunking document for Neo4j...")
            try:
                # Add timeout for chunking (5 minutes per file)
                nodes, edges = await asyncio.wait_for(
                    neo4j_chunker.chunk_for_neo4j(
                        text=text,
                        doc_id=doc_id,
                        doc_type=doc_type,
                        metadata=metadata,
                    ),
                    timeout=300.0  # 5 minutes timeout
                )
                print(f"✅ Generated {len(nodes)} nodes and {len(edges)} edges")
            except asyncio.TimeoutError:
                print(f"⏱️  Timeout: Chunking took too long for {filename}, skipping...")
                failed_files += 1
                continue
            except Exception as e:
                print(f"❌ Chunking error for {filename}: {e}")
                failed_files += 1
                continue

            # Display statistics
            node_types = {}
            for node in nodes:
                label = node["label"]
                node_types[label] = node_types.get(label, 0) + 1

            edge_types = {}
            for edge in edges:
                relation = edge["relation"]
                edge_types[relation] = edge_types.get(relation, 0) + 1

            print("\n📊 Nodes by type:")
            for label, count in sorted(node_types.items()):
                print(f"   - {label}: {count}")

            print("\n📊 Edges by relationship type:")
            for relation, count in sorted(edge_types.items()):
                print(f"   - {relation}: {count}")

            print()

            # Step 5: Insert into Neo4j
            print("💾 Step 5: Inserting into Neo4j...")
            
            # Insert nodes
            inserted_nodes = 0
            for node in nodes:
                # Merge properties with identifier
                all_props = {**node["properties"]}
                # Add identifier to properties if not already present
                id_prop = node.get("identifier_prop", "id")
                if id_prop not in all_props:
                    all_props[id_prop] = node["identifier"]

                try:
                    neo4j.upsert_node(
                        label=node["label"],
                        node_id=node["identifier"],
                        properties=all_props,
                    )
                    inserted_nodes += 1
                except Exception as e:
                    print(f"⚠️  Error inserting node {node['label']}:{node['identifier']}: {e}")

            print(f"✅ Inserted {inserted_nodes}/{len(nodes)} nodes")

            # Insert edges
            inserted_edges = 0
            for edge in edges:
                try:
                    neo4j.upsert_edge(
                        src_label=edge["source_label"],
                        src_id=edge["source_id"],
                        relation=edge["relation"],
                        tgt_label=edge["target_label"],
                        tgt_id=edge["target_id"],
                        properties=edge.get("properties", {}),
                    )
                    inserted_edges += 1
                except Exception as e:
                    print(f"⚠️  Error inserting edge {edge['relation']}: {e}")

            print(f"✅ Inserted {inserted_edges}/{len(edges)} edges")
            print()

            all_inserted_nodes += inserted_nodes
            all_inserted_edges += inserted_edges
            successful_files += 1

            # Step 6: Query and verify
            print("🔍 Step 6: Querying and verifying data...")
            
            # Query document node
            with neo4j._driver.session() as session:
                result = session.run(
                    """
                    MATCH (d:Document {doc_id: $doc_id})
                    RETURN d
                    """,
                    {"doc_id": doc_id},
                )
                doc_record = result.single()
                if doc_record:
                    doc_node = doc_record["d"]
                    print(f"✅ Document node found: {doc_node.get('title', doc_id)}")
                    print(f"   - doc_type: {doc_node.get('doc_type')}")
                    print(f"   - agency: {doc_node.get('agency')}")
                    print(f"   - effect_date: {doc_node.get('effective_date')}")
                    print(f"   - field: {doc_node.get('field')}")
                    print(f"   - status: {doc_node.get('status')}")
                else:
                    print(f"❌ Document node not found: {doc_id}")

                # Count articles
                result = session.run(
                    """
                    MATCH (d:Document {doc_id: $doc_id})-[:HAS_ARTICLE]->(a:Article)
                    RETURN count(a) as count
                    """,
                    {"doc_id": doc_id},
                )
                single_record = result.single()
                article_count = single_record["count"] if single_record else 0
                print(f"✅ Articles: {article_count}")

                # Count clauses
                result = session.run(
                    """
                    MATCH (d:Document {doc_id: $doc_id})-[:HAS_ARTICLE]->(a:Article)-[:HAS_CLAUSE]->(c:Clause)
                    RETURN count(c) as count
                    """,
                    {"doc_id": doc_id},
                )
                single_record = result.single()
                clause_count = single_record["count"] if single_record else 0
                print(f"✅ Clauses: {clause_count}")

                # Count relationships
                result = session.run(
                    """
                    MATCH (d:Document {doc_id: $doc_id})-[r]-()
                    RETURN type(r) as relation, count(r) as count
                    ORDER BY count DESC
                    """,
                    {"doc_id": doc_id},
                )
                print("\n📊 Relationships:")
                for record in result:
                    print(f"   - {record['relation']}: {record['count']}")

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
    print(f"📊 Total nodes inserted: {all_inserted_nodes}")
    print(f"📊 Total edges inserted: {all_inserted_edges}")
    print()

    # Database statistics
    print("🔍 Database Statistics:")
    try:
        with neo4j._driver.session() as session:
            # Count nodes by type
            result = session.run(
                """
                MATCH (n)
                RETURN labels(n)[0] as label, count(n) as count
                ORDER BY count DESC
                """
            )
            print("\nNodes by type:")
            for record in result:
                print(f"   - {record['label']}: {record['count']}")

            # Count relationships by type
            result = session.run(
                """
                MATCH ()-[r]->()
                RETURN type(r) as relation, count(r) as count
                ORDER BY count DESC
                """
            )
            print("\nRelationships by type:")
            for record in result:
                print(f"   - {record['relation']}: {record['count']}")

            # Count documents
            result = session.run(
                """
                MATCH (d:Document)
                RETURN count(d) as count
                """
            )
            single_record = result.single()
            doc_count = single_record["count"] if single_record else 0
            print(f"\n✅ Total documents in database: {doc_count}")

    except Exception as e:
        print(f"⚠️  Could not query database statistics: {e}")

    print(f"\n{'=' * 70}")
    print("✅ End-to-end test completed!")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    asyncio.run(test_parser_to_neo4j_flow())

