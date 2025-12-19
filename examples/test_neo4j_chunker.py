"""
Test script for Neo4jChunker.

This script demonstrates how to use Neo4jChunker to:
1. Parse Vietnamese legal documents
2. Extract nodes (Document, Article, Clause, Term, Agency)
3. Extract relationships (HAS_ARTICLE, HAS_CLAUSE, CITES, etc.)
4. Insert data into Neo4j database
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from dotenv import load_dotenv

from vihermes.config.settings import get_settings
from vihermes.lawgraph.neo4j_client import Neo4jClient
from vihermes.preprocess.neo4j_chunker import Neo4jChunker
from vihermes.preprocess.parser import DocumentParser
from vihermes.preprocess.models import DocumentMetadata

load_dotenv()


async def test_neo4j_chunker():
    """Test Neo4jChunker with sample documents."""
    print("=" * 70)
    print("Testing: Neo4jChunker")
    print("=" * 70)
    print()

    settings = get_settings()

    # Setup Neo4j
    print("🔧 Setting up Neo4j...")
    neo4j = Neo4jClient(
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password,
    )
    neo4j.init_schema()
    print("✅ Neo4j initialized\n")

    # Setup chunker and parser
    parser = DocumentParser()
    neo4j_chunker = Neo4jChunker()

    # Get project root
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"

    # Test files with metadata
    test_files = [
        {
            "file": "sample_law_1.txt",
            "doc_id": "L-2019-38",
            "doc_type": "Law",
            "metadata": DocumentMetadata(
                document_id="L-2019-38",
                issuing_authority="Quốc hội",
                effect_date="2019-07-01",
                field="Thuế",
                status="effective",
                source_url="https://example.com/law-38-2019",
            ),
        },
        {
            "file": "sample_law_2.txt",
            "doc_id": "NĐ-2020-126",
            "doc_type": "Decree",
            "metadata": DocumentMetadata(
                document_id="NĐ-2020-126",
                issuing_authority="Chính phủ",
                effect_date="2020-10-19",
                field="Thuế",
                status="effective",
                source_url="https://example.com/decree-126-2020",
            ),
        },
    ]

    for test_case in test_files:
        filename = test_case["file"]
        doc_id = test_case["doc_id"]
        doc_type = test_case["doc_type"]
        metadata = test_case["metadata"]
        file_path = data_dir / filename

        if not file_path.exists():
            print(f"❌ File not found: {file_path}")
            continue

        print(f"\n{'─' * 70}")
        print(f"📄 Document: {doc_id} ({doc_type})")
        print(f"📁 File: {filename}")
        print(f"{'─' * 70}")

        try:
            # Parse document with metadata extraction
            text, extracted_metadata = parser.parse_with_metadata(file_path)
            print(f"✅ Parsed {len(text)} characters")
            
            # Use extracted metadata if available, otherwise use provided metadata
            if extracted_metadata:
                print(f"✅ Extracted metadata from file:")
                print(f"   - document_id: {extracted_metadata.document_id}")
                print(f"   - issuing_authority: {extracted_metadata.issuing_authority}")
                print(f"   - effect_date: {extracted_metadata.effect_date}")
                print(f"   - field: {extracted_metadata.field}")
                print(f"   - status: {extracted_metadata.status}")
                # Merge extracted metadata with provided metadata
                # Prefer extracted metadata for fields that are filled
                if extracted_metadata.issuing_authority:
                    metadata.issuing_authority = extracted_metadata.issuing_authority
                if extracted_metadata.effect_date:
                    metadata.effect_date = extracted_metadata.effect_date
                if extracted_metadata.field:
                    metadata.field = extracted_metadata.field
                if extracted_metadata.status:
                    metadata.status = extracted_metadata.status
                if extracted_metadata.document_id and extracted_metadata.document_id != "UNKNOWN":
                    metadata.document_id = extracted_metadata.document_id
                    doc_id = extracted_metadata.document_id
            else:
                print("⚠️  No metadata found in file, using provided metadata")
            print()

            # Chunk for Neo4j
            print(f"{'=' * 70}")
            print("🔗 Chunking for Neo4j...")
            print(f"{'=' * 70}")
            nodes, edges = await neo4j_chunker.chunk_for_neo4j(
                text=text,
                doc_id=doc_id,
                doc_type=doc_type,
                metadata=metadata,
            )
            print(f"✅ Generated {len(nodes)} nodes and {len(edges)} edges\n")

            # Display nodes by type
            print("📊 Nodes by type:")
            node_types = {}
            for node in nodes:
                label = node["label"]
                node_types[label] = node_types.get(label, 0) + 1
            for label, count in sorted(node_types.items()):
                print(f"  - {label}: {count}")

            # Display edges by relationship type
            print("\n📊 Edges by relationship type:")
            edge_types = {}
            for edge in edges:
                relation = edge["relation"]
                edge_types[relation] = edge_types.get(relation, 0) + 1
            for relation, count in sorted(edge_types.items()):
                print(f"  - {relation}: {count}")

            # Display sample nodes
            print("\n📋 Sample nodes:")
            for node in nodes[:5]:
                print(f"  - {node['label']}: {node['identifier']}")
                print(f"    Properties: {list(node['properties'].keys())}")

            # Display sample edges
            print("\n📋 Sample edges:")
            for edge in edges[:5]:
                print(
                    f"  - ({edge['source_label']}:{edge['source_id']}) "
                    f"-[:{edge['relation']}]-> "
                    f"({edge['target_label']}:{edge['target_id']})"
                )

            # Insert into Neo4j
            print(f"\n{'=' * 70}")
            print("💾 Inserting into Neo4j...")
            print(f"{'=' * 70}")

            for node in nodes:
                # Merge properties with identifier
                all_props = {**node["properties"]}
                # Add identifier to properties if not already present
                id_prop = node.get("identifier_prop", "id")
                if id_prop not in all_props:
                    all_props[id_prop] = node["identifier"]

                neo4j.upsert_node(
                    label=node["label"],
                    node_id=node["identifier"],
                    properties=all_props,
                )
            print(f"✅ Inserted {len(nodes)} nodes into Neo4j")

            for edge in edges:
                neo4j.upsert_edge(
                    src_label=edge["source_label"],
                    src_id=edge["source_id"],
                    relation=edge["relation"],
                    tgt_label=edge["target_label"],
                    tgt_id=edge["target_id"],
                    properties=edge.get("properties", {}),
                )
            print(f"✅ Inserted {len(edges)} edges into Neo4j\n")

            # Display summary
            print("📊 Summary:")
            print(f"  - Nodes: {len(nodes)}")
            print(f"  - Edges: {len(edges)}")
            print(f"  - Node types: {len(node_types)}")
            print(f"  - Relationship types: {len(edge_types)}")
            print()

        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")
            import traceback

            traceback.print_exc()

    print(f"\n{'=' * 70}")
    print("✅ Test completed! Data stored in Neo4j")
    print(f"{'=' * 70}")

    # Query example: Count nodes by type
    print("\n📊 Database Statistics:")
    try:
        with neo4j._driver.session() as session:
            result = session.run(
                """
                MATCH (n)
                RETURN labels(n)[0] as label, count(n) as count
                ORDER BY count DESC
                """
            )
            for record in result:
                print(f"  - {record['label']}: {record['count']}")

            result = session.run(
                """
                MATCH ()-[r]->()
                RETURN type(r) as relation, count(r) as count
                ORDER BY count DESC
                """
            )
            print("\nRelationship counts:")
            for record in result:
                print(f"  - {record['relation']}: {record['count']}")
    except Exception as e:
        print(f"⚠️  Could not query statistics: {e}")


if __name__ == "__main__":
    asyncio.run(test_neo4j_chunker())

