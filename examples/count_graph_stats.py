"""
Script để đếm số nodes, số relationships và tính tổng tokens từ toàn bộ entities và relations trong Neo4j graph.

Thông tin thống kê:
- Tổng số nodes (entities)
- Tổng số relationships
- Tổng số tokens từ tất cả properties của nodes
- Tổng số tokens từ tất cả properties của relationships
- Chi tiết theo từng loại node và relationship
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

from dotenv import load_dotenv

from uraxlaw.config.settings import get_settings
from uraxlaw.lawgraph.neo4j_client import Neo4jClient, NODE_LABELS, REL_TYPES

load_dotenv()


def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """
    Đếm số tokens từ text.
    Nếu tiktoken không có sẵn, sẽ đếm số words (approximation).
    """
    if not text or not isinstance(text, str):
        return 0
    
    if TIKTOKEN_AVAILABLE:
        try:
            encoding = tiktoken.get_encoding(encoding_name)
            return len(encoding.encode(text))
        except Exception:
            # Fallback to word count if encoding fails
            return len(text.split())
    else:
        # Fallback: đếm words (rough approximation)
        return len(text.split())


def extract_text_from_properties(props: Dict[str, Any]) -> str:
    """Trích xuất tất cả text từ properties của node/relationship."""
    texts = []
    for key, value in props.items():
        if value is None:
            continue
        if isinstance(value, str):
            texts.append(value)
        elif isinstance(value, (int, float)):
            texts.append(str(value))
        elif isinstance(value, (list, dict)):
            texts.append(json.dumps(value, ensure_ascii=False))
    return " ".join(texts)


def get_all_nodes(neo4j_client: Neo4jClient) -> List[Dict[str, Any]]:
    """Lấy tất cả nodes từ Neo4j."""
    query = """
    MATCH (n)
    RETURN labels(n) as labels, properties(n) as props
    """
    results = neo4j_client.run_cypher(query)
    return results


def get_all_relationships(neo4j_client: Neo4jClient) -> List[Dict[str, Any]]:
    """Lấy tất cả relationships từ Neo4j cùng với thông tin source và target nodes."""
    query = """
    MATCH (a)-[r]->(b)
    RETURN type(r) as rel_type, 
           properties(r) as props,
           labels(a) as src_labels,
           labels(b) as tgt_labels,
           a.id as src_id,
           a.doc_id as src_doc_id,
           a.article_id as src_article_id,
           a.clause_id as src_clause_id,
           a.title as src_title,
           a.name as src_name,
           b.id as tgt_id,
           b.doc_id as tgt_doc_id,
           b.article_id as tgt_article_id,
           b.clause_id as tgt_clause_id,
           b.title as tgt_title,
           b.name as tgt_name
    """
    results = neo4j_client.run_cypher(query)
    return results


def count_nodes_by_label(neo4j_client: Neo4jClient) -> Dict[str, int]:
    """Đếm số nodes theo từng label."""
    counts = {}
    for label in NODE_LABELS:
        query = f"""
        MATCH (n:{label})
        RETURN count(n) as count
        """
        result = neo4j_client.run_cypher(query)
        if result:
            counts[label] = result[0].get("count", 0)
        else:
            counts[label] = 0
    return counts


def count_relationships_by_type(neo4j_client: Neo4jClient) -> Dict[str, int]:
    """Đếm số relationships theo từng type."""
    counts = {}
    for rel_type in REL_TYPES:
        query = f"""
        MATCH ()-[r:{rel_type}]->()
        RETURN count(r) as count
        """
        result = neo4j_client.run_cypher(query)
        if result:
            counts[rel_type] = result[0].get("count", 0)
        else:
            counts[rel_type] = 0
    return counts


def calculate_tokens_for_nodes(neo4j_client: Neo4jClient) -> tuple[int, Dict[str, int]]:
    """Tính tổng tokens từ tất cả nodes và tokens theo từng label."""
    all_nodes = get_all_nodes(neo4j_client)
    total_tokens = 0
    tokens_by_label: Dict[str, int] = {}
    
    for node in all_nodes:
        labels = node.get("labels", [])
        props = node.get("props", {})
        
        # Lấy primary label
        primary_label = labels[0] if labels else "Unknown"
        for lbl in labels:
            if lbl in NODE_LABELS:
                primary_label = lbl
                break
        
        # Trích xuất text từ properties
        text = extract_text_from_properties(props)
        tokens = count_tokens(text)
        
        total_tokens += tokens
        
        # Đếm theo label
        if primary_label not in tokens_by_label:
            tokens_by_label[primary_label] = 0
        tokens_by_label[primary_label] += tokens
    
    return total_tokens, tokens_by_label


def calculate_tokens_for_relationships(neo4j_client: Neo4jClient) -> tuple[int, Dict[str, int]]:
    """
    Tính tổng tokens từ tất cả relationships và tokens theo từng type.
    Tính tokens từ:
    - Relationship type name
    - Relationship properties (nếu có)
    - Source và target node identifiers và titles (để đếm đầy đủ thông tin của relationships)
    """
    all_rels = get_all_relationships(neo4j_client)
    total_tokens = 0
    tokens_by_type: Dict[str, int] = {}
    
    for rel in all_rels:
        rel_type = rel.get("rel_type", "Unknown")
        props = rel.get("props", {})
        
        # Tạo text từ relationship type name (luôn có)
        texts = []
        if rel_type:
            texts.append(str(rel_type))
        
        # Thêm properties nếu có
        if props:
            props_text = extract_text_from_properties(props)
            if props_text and props_text.strip():
                texts.append(props_text)
        
        # Thêm source node identifiers và titles (tính vào relationship tokens)
        src_texts = []
        for key in ["src_id", "src_doc_id", "src_article_id", "src_clause_id", "src_title", "src_name"]:
            value = rel.get(key)
            if value and str(value).strip():
                src_texts.append(str(value).strip())
        if src_texts:
            texts.extend(src_texts)
        
        # Thêm target node identifiers và titles (tính vào relationship tokens)
        tgt_texts = []
        for key in ["tgt_id", "tgt_doc_id", "tgt_article_id", "tgt_clause_id", "tgt_title", "tgt_name"]:
            value = rel.get(key)
            if value and str(value).strip():
                tgt_texts.append(str(value).strip())
        if tgt_texts:
            texts.extend(tgt_texts)
        
        # Thêm source và target labels
        src_labels = rel.get("src_labels", [])
        if src_labels:
            for label in src_labels:
                if label and str(label).strip():
                    texts.append(str(label).strip())
        
        tgt_labels = rel.get("tgt_labels", [])
        if tgt_labels:
            for label in tgt_labels:
                if label and str(label).strip():
                    texts.append(str(label).strip())
        
        # Tính tokens từ tất cả text đã thu thập (lọc bỏ empty strings)
        texts_filtered = [t for t in texts if t and t.strip()]
        combined_text = " ".join(texts_filtered) if texts_filtered else str(rel_type) if rel_type else ""
        tokens = count_tokens(combined_text) if combined_text else 0
        
        total_tokens += tokens
        
        # Đếm theo type
        if rel_type not in tokens_by_type:
            tokens_by_type[rel_type] = 0
        tokens_by_type[rel_type] += tokens
    
    return total_tokens, tokens_by_type


def main() -> None:
    """Main function để chạy thống kê."""
    print("=" * 70)
    print("📊 Graph Statistics - Neo4j Database")
    print("=" * 70)
    print()
    
    # Token counting method info
    if not TIKTOKEN_AVAILABLE:
        print("⚠️  Lưu ý: tiktoken không có sẵn, sẽ dùng cách đếm words đơn giản")
        print()
    
    # Kết nối Neo4j
    print("🔧 Đang kết nối với Neo4j...")
    settings = get_settings()
    neo4j_client = Neo4jClient(
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password,
    )
    
    try:
        # Đếm số nodes
        print("📊 Đang đếm nodes...")
        node_counts = count_nodes_by_label(neo4j_client)
        total_nodes = sum(node_counts.values())
        
        # Đếm số relationships
        print("📊 Đang đếm relationships...")
        rel_counts = count_relationships_by_type(neo4j_client)
        total_rels = sum(rel_counts.values())
        
        # Tính tokens cho nodes
        print("🔢 Đang tính tokens cho nodes...")
        node_tokens_total, node_tokens_by_label = calculate_tokens_for_nodes(neo4j_client)
        
        # Tính tokens cho relationships
        print("🔢 Đang tính tokens cho relationships...")
        rel_tokens_total, rel_tokens_by_type = calculate_tokens_for_relationships(neo4j_client)
        
        # Debug: Kiểm tra một vài relationships mẫu
        if total_rels > 0 and rel_tokens_total == 0:
            print(f"⚠️  Cảnh báo: Có {total_rels} relationships nhưng tổng tokens = 0")
            print("   Đang kiểm tra mẫu relationships...")
            sample_rels = get_all_relationships(neo4j_client)[:5]
            for i, rel in enumerate(sample_rels):
                print(f"   Relationship {i+1}: type={rel.get('rel_type')}, props={rel.get('props')}")
        
        # Tổng tokens
        total_tokens = node_tokens_total + rel_tokens_total
        
        # In kết quả
        print()
        print("=" * 70)
        print("📈 KẾT QUẢ THỐNG KÊ")
        print("=" * 70)
        print()
        
        print(f"🔹 Tổng số Nodes: {total_nodes:,}")
        print(f"🔹 Tổng số Relationships: {total_rels:,}")
        print(f"🔹 Tổng số Tokens: {total_tokens:,}")
        print(f"   ├─ Tokens từ Nodes: {node_tokens_total:,}")
        print(f"   └─ Tokens từ Relationships: {rel_tokens_total:,}")
        print()
        
        # Chi tiết theo label
        print("=" * 70)
        print("📋 CHI TIẾT NODES THEO LABEL")
        print("=" * 70)
        print(f"{'Label':<25} {'Số lượng':<15} {'Tokens':<15}")
        print("-" * 70)
        for label in sorted(node_counts.keys(), key=lambda x: node_counts[x], reverse=True):
            count = node_counts[label]
            tokens = node_tokens_by_label.get(label, 0)
            if count > 0:
                print(f"{label:<25} {count:<15,} {tokens:<15,}")
        print()
        
        # Chi tiết theo relationship type
        print("=" * 70)
        print("📋 CHI TIẾT RELATIONSHIPS THEO TYPE")
        print("=" * 70)
        print(f"{'Type':<30} {'Số lượng':<15} {'Tokens':<15}")
        print("-" * 70)
        for rel_type in sorted(rel_counts.keys(), key=lambda x: rel_counts[x], reverse=True):
            count = rel_counts[rel_type]
            tokens = rel_tokens_by_type.get(rel_type, 0)
            if count > 0:
                print(f"{rel_type:<30} {count:<15,} {tokens:<15,}")
        print()
        
        # Summary
        print("=" * 70)
        print("📊 SUMMARY")
        print("=" * 70)
        print(f"Total Nodes: {total_nodes:,}")
        print(f"Total Relationships: {total_rels:,}")
        print(f"Total Tokens: {total_tokens:,}")
        if total_tokens > 0:
            print(f"  - From Nodes: {node_tokens_total:,} ({node_tokens_total/total_tokens*100:.2f}%)")
            print(f"  - From Relationships: {rel_tokens_total:,} ({rel_tokens_total/total_tokens*100:.2f}%)")
        else:
            print(f"  - From Nodes: {node_tokens_total:,}")
            print(f"  - From Relationships: {rel_tokens_total:,}")
        print()
        
        # Token counting method info
        if TIKTOKEN_AVAILABLE:
            print("ℹ️  Sử dụng tiktoken (cl100k_base) để đếm tokens")
        else:
            print("⚠️  Tiktoken không có sẵn, đang dùng cách đếm words (approximation)")
        print()
        
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        import traceback
        traceback.print_exc()
    finally:
        neo4j_client.close()
        print("✅ Đã đóng kết nối Neo4j")


if __name__ == "__main__":
    main()

