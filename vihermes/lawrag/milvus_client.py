import random
from types import SimpleNamespace
from typing import Callable, Dict, List, Optional
from pymilvus import (
    MilvusClient as PyMilvusClient,
    CollectionSchema,
    FieldSchema,
    DataType
)

class MilvusSchemaManager:
    """
    A class to manage schema creation and setup for a Milvus Collection
    dedicated to Graph RAG.
    
    This class now includes methods for:
    - Connection
    - Schema/Collection/Index creation
    - Data insertion
    - Hybrid searching
    - Deletion by document ID
    """
    
    def __init__(self, collection_name, dense_dim, milvus_uri="http://localhost:19530"):
        """
        Initializes the manager.
        
        Args:
            collection_name (str): The name of the collection to be created.
            dense_dim (int): The dimension of the dense vector (e.g., 768).
            milvus_uri (str): The URI to connect to Milvus.
        """
        self.collection_name = collection_name
        self.dense_dim = dense_dim
        self.milvus_uri = milvus_uri
        self.client = None
        print(f"Initializing MilvusSchemaManager for collection '{collection_name}' with dim={dense_dim}.")

    def connect(self):
        """
        Establishes a connection to Milvus.
        """
        print(f"Connecting to Milvus at {self.milvus_uri}...")
        try:
            self.client = PyMilvusClient(uri=self.milvus_uri)
            print("Milvus connection successful.")
            return True
        except Exception as e:
            print(f"ERROR: Could not connect to Milvus. Ensure Milvus (standalone or cluster) is running at {self.milvus_uri}")
            print(f"Error details: {e}")
            self.client = None
            return False

    def close(self):
        """
        Closes the connection to Milvus.
        """
        if self.client:
            self.client.close()
            print("Closed Milvus connection.")
            self.client = None

    def _define_schema(self):
        """(Internal) Defines the schema structure."""
        print("Defining schema...")
        fields = [
            # Primary key
            FieldSchema(name="article_id", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=64),
            
            # Document identifiers
            FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="doc_type", dtype=DataType.VARCHAR, max_length=20),
            
            # Article/Clause structure
            FieldSchema(name="article_number", dtype=DataType.INT64),
            FieldSchema(name="clause_number", dtype=DataType.VARCHAR, max_length=10),
            
            # Content
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=1000),  # Increased from 255 to 1000
            
            # Vector embeddings
            FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=self.dense_dim),
            FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
            
            # Metadata
            FieldSchema(name="year", dtype=DataType.INT64),
            FieldSchema(name="agency", dtype=DataType.VARCHAR, max_length=500),  # Increased from 255 to 500
            FieldSchema(name="status", dtype=DataType.VARCHAR, max_length=20),
            FieldSchema(name="effective_date", dtype=DataType.VARCHAR, max_length=20),  # ISO date string
            FieldSchema(name="source_url", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="last_update", dtype=DataType.INT64),  # Unix timestamp
        ]
        return CollectionSchema(
            fields=fields,
            description="Schema for Vietnamese legal documents with article-level chunking",
            enable_dynamic_field=False
        )

    def _create_indexes(self):
        """(Internal) Creates the necessary indexes for searching."""
        if not self.client:
            print("ERROR: Client is not connected. Cannot create index.")
            return

        # 1. Create Index for Dense Vector
        print("Creating index for 'dense_vector' field...")
        dense_index_params = self.client.prepare_index_params()
        dense_index_params.add_index(
            field_name="dense_vector",
            index_type="HNSW",
            metric_type="COSINE",
            params={"M": 16, "efConstruction": 256}
        )
        self.client.create_index(
            collection_name=self.collection_name,
            index_params=dense_index_params
        )
        print("Successfully created 'dense_vector' index (HNSW, COSINE).")

        # 2. Create index for Sparse Vector
        print("Creating index for 'sparse_vector' field...")
        sparse_index_params = self.client.prepare_index_params()
        sparse_index_params.add_index(
            field_name="sparse_vector",
            index_type="SPARSE_INVERTED_INDEX", 
            metric_type="IP", 
            params={"drop_ratio_build": 0.1}
        )
        self.client.create_index(
            collection_name=self.collection_name,
            index_params=sparse_index_params
        )
        print("Successfully created 'sparse_vector' index (SPARSE_INVERTED_INDEX, IP).")
        
        # 3. Create scalar indexes for numeric fields only
        # Note: STL_SORT only supports numeric fields (INT64, FLOAT, etc.)
        # VARCHAR fields (doc_id, doc_type, status) don't need indexes for filtering
        print("Creating scalar indexes for numeric fields...")
        numeric_indexes = [
            ("article_number", "article_number_idx"),
            ("year", "year_idx"),
        ]
        for field_name, index_name in numeric_indexes:
            try:
                scalar_index_params = self.client.prepare_index_params()
                scalar_index_params.add_index(
                    field_name=field_name,
                    index_type="STL_SORT"
                )
                self.client.create_index(
                    collection_name=self.collection_name,
                    index_params=scalar_index_params
                )
                print(f"Successfully created scalar index '{index_name}' for '{field_name}'.")
            except Exception as e:
                print(f"Warning: Could not create index for '{field_name}': {e}")
        
        # Note: VARCHAR fields (doc_id, doc_type, status) can be filtered without indexes
        # If needed, you can use MARISA trie index for VARCHAR fields, but it's optional

    def recreate_collection(self):
        """
        Deletes the old collection (if it exists) and creates a new one,
        including index creation and loading into memory.
        """
        if not self.client:
            print("ERROR: Client is not connected. Please call .connect() first.")
            return

        if self.client.has_collection(collection_name=self.collection_name):
            print(f"Found existing collection '{self.collection_name}'. Dropping...")
            self.client.drop_collection(collection_name=self.collection_name)
            print(f"Dropped collection '{self.collection_name}'.")
        else:
            print(f"Collection '{self.collection_name}' does not exist, will create a new one.")

        schema = self._define_schema()
        print(f"Creating collection '{self.collection_name}'...")
        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            consistency_level="Bounded"
        )
        print(f"Successfully created collection '{self.collection_name}'.")

        self._create_indexes()
        
        print(f"Loading collection '{self.collection_name}' into memory for searching...")
        self.client.load_collection(collection_name=self.collection_name)
        print(f"Complete! Collection '{self.collection_name}' is ready to use.")

    def insert(self, data):
        """
        Inserts data into the collection.
        
        Args:
            data (list[dict]): A list of dictionaries, where each dictionary
                               matches the schema.
        """
        if not self.client:
            print("ERROR: Client is not connected. Cannot insert.")
            return None
            
        print(f"Inserting {len(data)} records...")
        try:
            res = self.client.insert(
                collection_name=self.collection_name,
                data=data
            )
            print("Insertion successful.")
            return res
        except Exception as e:
            print(f"ERROR during insertion: {e}")
            return None

    def flush(self):
        """
        Flushes the collection to make inserted data searchable.
        """
        if not self.client:
            print("ERROR: Client is not connected. Cannot flush.")
            return
        print("Flushing collection...")
        try:
            # Newer PyMilvus MilvusClient exposes `flush`
            if hasattr(self.client, "flush"):
                self.client.flush(collection_name=self.collection_name)
            else:
                # Many operations are auto-flushed; proceed without hard flush
                print("flush() not available; skipping explicit flush.")
        except Exception as e:
            print(f"Flush not supported or failed: {e}. Continuing.")
        else:
            print("Flush complete. Data is now searchable.")

    def search(self, dense_query_vector, sparse_query_vector, limit=5, output_fields=None):
        """
        Performs a hybrid search using both dense and sparse vectors.
        
        Args:
            dense_query_vector (list[float]): The query vector from the dense model.
            sparse_query_vector (dict[int, float]): The query vector from the sparse model.
            limit (int): The final number of results to return after reranking.
            output_fields (list[str]): List of fields to return in the results.
        
        Returns:
            list[dict]: A list of search results.
        """
        if not self.client:
            print("ERROR: Client is not connected. Cannot search.")
            return []

        if output_fields is None:
            output_fields = ["article_id", "doc_id", "text", "title", "doc_type", "article_number", "clause_number"]

        req_dense = {
            "data": [dense_query_vector],
            "anns_field": "dense_vector",
            "param": {"metric_type": "COSINE", "params": {"ef": 128}},
            "limit": limit
        }
        req_sparse = {
            "data": [sparse_query_vector],
            "anns_field": "sparse_vector",
            "param": {"metric_type": "IP"},
            "limit": limit
        }
        
        search_requests = [req_dense, req_sparse]
        rerank = {"strategy": "rrf", "k": 60}
        
        print(f"Performing hybrid search with RRF reranking (limit={limit})...")
        try:
            results = self.client.search(
                collection_name=self.collection_name,
                data=search_requests,
                rerank=rerank,
                limit=limit,
                output_fields=output_fields
            )
            return results[0]
        except Exception as e:
            print(f"ERROR during search: {e}")
            return []

    # --- PHƯƠNG THỨC MỚI ---
    def delete_by_doc_id(self, doc_id):
        """
        Deletes all entities (chunks) associated with a specific doc_id.
        
        Args:
            doc_id (str): The document ID to delete all chunks for.
        """
        if not self.client:
            print("ERROR: Client is not connected. Cannot delete.")
            return None
        
        # Build the filter expression
        # Note: Use double quotes inside the string for string literals in the filter
        expression = f"doc_id == \"{doc_id}\""
        
        print(f"Attempting to delete entities with filter: {expression}")
        
        try:
            # First, query to see how many entities will be deleted
            # We set a high limit just to get a count, 
            # Milvus doesn't return all results by default but count is what we need
            query_res = self.client.query(
                collection_name=self.collection_name,
                filter=expression,
                output_fields=["id"] 
            )
            num_to_delete = len(query_res)
            
            if num_to_delete == 0:
                print(f"No entities found with doc_id '{doc_id}'. Nothing to delete.")
                return
            
            print(f"Found {num_to_delete} entities. Proceeding with deletion...")

            # Perform the delete operation
            res = self.client.delete(
                collection_name=self.collection_name,
                filter=expression
            )
            
            print(f"Deletion request successful. Primary keys of deleted entities: {res}")
            
            # IMPORTANT: Call flush after delete to make the change persistent
            self.flush()
            print(f"Flushed collection. Deletion of '{doc_id}' chunks is now persistent.")
            
            return res
            
        except Exception as e:
            print(f"ERROR during deletion for '{doc_id}': {e}")
            return None


class MilvusClient:
    """
    Lightweight wrapper expected by the HybridRetriever.
    - Connects to Milvus using host/port/collection
    - Optionally accepts an embedder callable to turn text -> dense vector
    - search(query, top_k) returns a list of hits with .chunk and .score
    """

    def __init__(self, host: str = "localhost", port: int = 19530, collection: str = "vihermes_articles", dense_dim: int = 768, embedder: Optional[Callable[[str], List[float]]] = None):
        self._uri = f"http://{host}:{port}"
        self._collection = collection
        self._dense_dim = dense_dim
        self._embedder = embedder
        self._client: Optional[PyMilvusClient] = None
        self._sparse_encoder: Optional[Callable[[str], Dict[int, float]]] = None

    def set_embedder(self, embedder: Callable[[str], List[float]], auto_detect_dim: bool = True):
        """
        Set embedder function. Optionally auto-detect dimension from embedder.
        
        Args:
            embedder: Function that takes text and returns embedding vector
            auto_detect_dim: If True, automatically detect and update dense_dim from embedder
        """
        self._embedder = embedder
        if auto_detect_dim:
            # Auto-detect dimension by calling embedder with a test string
            try:
                test_vec = embedder("test")
                if isinstance(test_vec, list) and len(test_vec) > 0:
                    self._dense_dim = len(test_vec)
            except Exception:
                pass  # Keep original dense_dim if detection fails

    def set_sparse_encoder(self, encoder: Callable[[str], Dict[int, float]]):
        self._sparse_encoder = encoder

    def _ensure_client(self) -> PyMilvusClient:
        if self._client is None:
            self._client = PyMilvusClient(uri=self._uri)
        return self._client

    def search(self, query: str, top_k: int = 5):
        client = self._ensure_client()
        if self._embedder is None and self._sparse_encoder is None:
            return []

        # Build dense vector if available
        dense_vec = None
        if self._embedder is not None:
            dense_vec = self._embedder(query)
            if not isinstance(dense_vec, list) or len(dense_vec) != self._dense_dim:
                raise ValueError("Embedder must return a list[float] of length dense_dim")

        # Build sparse vector if available
        sparse_vec = None
        if self._sparse_encoder is not None:
            sparse_raw = self._sparse_encoder(query)
            # Convert dict to Milvus sparse format: {"indices": [...], "values": [...]}
            if isinstance(sparse_raw, dict) and "indices" in sparse_raw and "values" in sparse_raw:
                sparse_vec = sparse_raw
            elif isinstance(sparse_raw, dict):
                # Convert {index: value} to {"indices": [...], "values": [...]}
                items = sorted(sparse_raw.items(), key=lambda x: x[0])
                sparse_vec = {"indices": [int(i) for i, _ in items], "values": [float(v) for _, v in items]}
            else:
                sparse_vec = sparse_raw

        try:
            # PyMilvus MilvusClient doesn't support hybrid search directly
            # Use dense-only if available (most common case)
            # For true hybrid search, use MilvusSchemaManager.search() instead
            if dense_vec is not None:
                # Dense-only search
                results = client.search(
                    collection_name=self._collection,
                    data=[dense_vec],
                    anns_field="dense_vector",
                    search_params={"metric_type": "COSINE", "params": {"ef": 128}},
                    limit=max(1, top_k),
                    output_fields=["article_id", "doc_id", "text", "title", "doc_type"],
                )
                hits = results[0] if results else []
            elif sparse_vec is not None:
                # Sparse-only search
                results = client.search(
                    collection_name=self._collection,
                    data=[sparse_vec],
                    anns_field="sparse_vector",
                    search_params={"metric_type": "IP"},
                    limit=max(1, top_k),
                    output_fields=["article_id", "doc_id", "text", "title", "doc_type"],
                )
                hits = results[0] if results else []
            else:
                hits = []
        except Exception as e:
            print(f"Milvus search error: {e}")
            hits = []

        from vihermes.lawrag.models import Chunk

        out = []
        for h in hits:
            try:
                # Try dict-like access first (most common case)
                try:
                    entity = h["entity"] if "entity" in h else {}
                    chunk_id = entity.get("article_id") if isinstance(entity, dict) else h.get("article_id")
                    doc_id = entity.get("doc_id", "") if isinstance(entity, dict) else ""
                    text = entity.get("text", "") if isinstance(entity, dict) else ""
                    title = entity.get("title", "") if isinstance(entity, dict) else h.get("title", "")
                    distance = h["distance"] if "distance" in h else h.get("distance")
                except (KeyError, TypeError):
                    # Fallback: attribute access
                    entity = getattr(h, "entity", {})
                    if isinstance(entity, dict):
                        chunk_id = entity.get("article_id")
                        doc_id = entity.get("doc_id", "")
                        text = entity.get("text", "")
                        title = entity.get("title", "")
                    else:
                        chunk_id = getattr(entity, "article_id", None) or getattr(h, "article_id", None)
                        doc_id = getattr(entity, "doc_id", "") or getattr(h, "doc_id", "")
                        text = getattr(entity, "text", "") or getattr(h, "text", "")
                        title = getattr(entity, "title", "") or getattr(h, "title", "")
                    distance = getattr(h, "distance", None)
                
                if chunk_id and text:
                    chunk = Chunk(
                        id=str(chunk_id),
                        document_id=str(doc_id),
                        node_id=None,
                        text=str(text),
                    )
                    # Convert distance to score (COSINE: distance = 1 - similarity)
                    if distance is not None:
                        score = 1.0 - float(distance) if distance <= 1.0 else 1.0 / (1.0 + float(distance))
                    else:
                        score = 0.0
                    result_obj = SimpleNamespace(chunk=chunk, score=float(score))
                    # Store title in result object for later access
                    result_obj.title = str(title) if title else ""
                    # Also store raw hit for debugging
                    result_obj.raw_hit = h
                    out.append(result_obj)
            except Exception:
                continue
        
        return out

# --- Example usage of the class ---
if __name__ == "__main__":
    
    # --- Configuration ---
    COLLECTION_NAME = "graph_rag_collection"
    DENSE_DIM = 768 # CHANGE THIS
    MILVUS_URI = "http://localhost:19530"
    
    print("--- Starting example script ---")
    
    manager = MilvusSchemaManager(
        collection_name=COLLECTION_NAME,
        dense_dim=DENSE_DIM,
        milvus_uri=MILVUS_URI
    )
    
    try:
        if manager.connect():
            # 1. Recreate the collection
            manager.recreate_collection()
            
            # 2. Prepare and Insert Mock Data
            print("\n--- Testing Data Insertion ---")
            mock_data = [
                # Example chunks with new schema
                {
                    "article_id": "L-2013-43:1",
                    "doc_id": "L-2013-43",
                    "doc_type": "Law",
                    "article_number": 1,
                    "clause_number": None,
                    "text": "This is a guide on JIRA.",
                    "title": "Phạm vi điều chỉnh",
                    "dense_vector": [random.random() for _ in range(DENSE_DIM)],
                    "sparse_vector": {100: 0.2, 500: 0.9},
                    "year": 2013,
                    "agency": "Quốc hội",
                    "status": "active",
                    "effective_date": "2013-01-01",
                    "source_url": "http://example.com/doc1",
                    "last_update": 1234567890,
                },
                {
                    "article_id": "L-2013-43:2.1",
                    "doc_id": "L-2013-43",
                    "doc_type": "Law",
                    "article_number": 2,
                    "clause_number": "1",
                    "text": "JIRA is good for bug tracking.",
                    "title": "Đối tượng áp dụng",
                    "dense_vector": [random.random() for _ in range(DENSE_DIM)],
                    "sparse_vector": {500: 0.7, 2000: 0.3},
                    "year": 2013,
                    "agency": "Quốc hội",
                    "status": "active",
                    "effective_date": "2013-01-01",
                    "source_url": "http://example.com/doc1",
                    "last_update": 1234567890,
                },
            ]
            manager.insert(mock_data)
            manager.flush()

            # 3. Test Search (Should find all 3 chunks)
            print("\n--- Testing Search (Before Delete) ---")
            # Query for "JIRA guide"
            search_results = manager.search(
                dense_query_vector=[random.random() for _ in range(DENSE_DIM)],
                sparse_query_vector={100: 0.6, 500: 0.8},
                limit=3,
                output_fields=["article_id", "doc_id", "text"]
            )
            print(f"Search found {len(search_results)} results:")
            for hit in search_results:
                entity = hit.get("entity", {})
                print(f"  Article ID: {entity.get('article_id')}, Doc ID: {entity.get('doc_id')}")

            # 4. Test Delete Function
            print("\n--- Testing Deletion ---")
            # Delete all chunks from "L-2013-43"
            manager.delete_by_doc_id("L-2013-43")

            # 5. Test Search (After Delete)
            print("\n--- Testing Search (After Delete) ---")
            # Same query for "JIRA guide"
            # It should now only find "chunk_003" or nothing related to JIRA
            search_results_after_delete = manager.search(
                dense_query_vector=[random.random() for _ in range(DENSE_DIM)],
                sparse_query_vector={100: 0.6, 500: 0.8}, # Still searching for JIRA
                limit=3,
                output_fields=["article_id", "doc_id", "text"]
            )
            
            print(f"Search found {len(search_results_after_delete)} results:")
            if search_results_after_delete:
                for hit in search_results_after_delete:
                    entity = hit.get("entity", {})
                    print(f"  Article ID: {entity.get('article_id')}, Doc ID: {entity.get('doc_id')}")
                
                # Check if any "L-2013-43" chunks were returned
                found_doc = any(hit.get("entity", {}).get("doc_id") == "L-2013-43" for hit in search_results_after_delete)
                if not found_doc:
                    print("SUCCESS: No chunks from 'L-2013-43' were found after deletion.")
                else:
                    print("FAILURE: Chunks from 'L-2013-43' were found after deletion.")
            else:
                print("SUCCESS: No results found (as expected, or unrelated results found).")
            
    except Exception as e:
        print(f"An unexpected error occurred in the main script: {e}")
    finally:
        # 6. Always close the connection
        manager.close()
        
    print("--- Example script finished ---")
