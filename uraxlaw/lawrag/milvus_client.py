import sys
import random
from pymilvus import (
    MilvusClient,
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
            self.client = MilvusClient(uri=self.milvus_uri)
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
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=256),
            FieldSchema(name="original_doc_id", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="url", dtype=DataType.VARCHAR, max_length=1024),
            FieldSchema(name="dense_vec", dtype=DataType.FLOAT_VECTOR, dim=self.dense_dim),
            FieldSchema(name="sparse_vec", dtype=DataType.SPARSE_FLOAT_VECTOR)
        ]
        return CollectionSchema(
            fields=fields,
            description="Schema for the Graph RAG system",
            enable_dynamic_field=False
        )

    def _create_indexes(self):
        """(Internal) Creates the necessary indexes for searching."""
        if not self.client:
            print("ERROR: Client is not connected. Cannot create index.")
            return

        # 1. Create Index for Dense Vector
        print("Creating index for 'dense_vec' field...")
        dense_index_params = self.client.prepare_index_params()
        dense_index_params.add_index(
            field_name="dense_vec",
            index_type="HNSW",
            metric_type="COSINE",
            params={"M": 16, "efConstruction": 256}
        )
        self.client.create_index(
            collection_name=self.collection_name,
            index_params=dense_index_params
        )
        print("Successfully created 'dense_vec' index (HNSW, COSINE).")

        # 2. Create index for Sparse Vector
        print("Creating index for 'sparse_vec' field...")
        sparse_index_params = self.client.prepare_index_params()
        sparse_index_params.add_index(
            field_name="sparse_vec",
            index_type="SPARSE_INVERTED_INDEX", 
            metric_type="IP", 
            params={"drop_ratio_build": 0.1}
        )
        self.client.create_index(
            collection_name=self.collection_name,
            index_params=sparse_index_params
        )
        print("Successfully created 'sparse_vec' index (SPARSE_INVERTED_INDEX, IP).")

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
        self.client.flush_collection(collection_name=self.collection_name)
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
            output_fields = ["id", "text", "source", "url"]

        req_dense = {
            "data": [dense_query_vector],
            "anns_field": "dense_vec",
            "param": {"metric_type": "COSINE", "params": {"ef": 128}},
            "limit": limit
        }
        req_sparse = {
            "data": [sparse_query_vector],
            "anns_field": "sparse_vec",
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
    def delete_by_original_doc_id(self, original_doc_id):
        """
        Deletes all entities (chunks) associated with a specific original_doc_id.
        
        Args:
            original_doc_id (str): The document ID to delete all chunks for.
        """
        if not self.client:
            print("ERROR: Client is not connected. Cannot delete.")
            return None
        
        # Build the filter expression
        # Note: Use double quotes inside the string for string literals in the filter
        expression = f"original_doc_id == \"{original_doc_id}\""
        
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
                print(f"No entities found with original_doc_id '{original_doc_id}'. Nothing to delete.")
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
            print(f"Flushed collection. Deletion of '{original_doc_id}' chunks is now persistent.")
            
            return res
            
        except Exception as e:
            print(f"ERROR during deletion for '{original_doc_id}': {e}")
            return None

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
                # Chunks for doc_1
                {"id": "chunk_001", "original_doc_id": "doc_1", "text": "This is a guide on JIRA.", "source": "Confluence", "url": "http://example.com/doc1", "dense_vec": [random.random() for _ in range(DENSE_DIM)], "sparse_vec": {100: 0.2, 500: 0.9}},
                {"id": "chunk_002", "original_doc_id": "doc_1", "text": "JIRA is good for bug tracking.", "source": "Confluence", "url": "http://example.com/doc1", "dense_vec": [random.random() for _ in range(DENSE_DIM)], "sparse_vec": {500: 0.7, 2000: 0.3}},
                # Chunk for doc_2
                {"id": "chunk_003", "original_doc_id": "doc_2", "text": "How to manage sprints.", "source": "Internal Wiki", "url": "http://example.com/doc3", "dense_vec": [random.random() for _ in range(DENSE_DIM)], "sparse_vec": {3000: 0.8}}
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
                output_fields=["id", "original_doc_id"]
            )
            print(f"Search found {len(search_results)} results:")
            for hit in search_results:
                print(f"  ID: {hit['id']}, DocID: {hit['entity']['original_doc_id']}")

            # 4. Test Delete Function
            print("\n--- Testing Deletion ---")
            # Delete all chunks from "doc_1"
            manager.delete_by_original_doc_id("doc_1")

            # 5. Test Search (After Delete)
            print("\n--- Testing Search (After Delete) ---")
            # Same query for "JIRA guide"
            # It should now only find "chunk_003" or nothing related to JIRA
            search_results_after_delete = manager.search(
                dense_query_vector=[random.random() for _ in range(DENSE_DIM)],
                sparse_query_vector={100: 0.6, 500: 0.8}, # Still searching for JIRA
                limit=3,
                output_fields=["id", "original_doc_id"]
            )
            
            print(f"Search found {len(search_results_after_delete)} results:")
            if search_results_after_delete:
                for hit in search_results_after_delete:
                    print(f"  ID: {hit['id']}, DocID: {hit['entity']['original_doc_id']}")
                
                # Check if any "doc_1" chunks were returned
                found_doc_1 = any(hit['entity']['original_doc_id'] == 'doc_1' for hit in search_results_after_delete)
                if not found_doc_1:
                    print("SUCCESS: No chunks from 'doc_1' were found after deletion.")
                else:
                    print("FAILURE: Chunks from 'doc_1' were found after deletion.")
            else:
                print("SUCCESS: No results found (as expected, or unrelated results found).")
            
    except Exception as e:
        print(f"An unexpected error occurred in the main script: {e}")
    finally:
        # 6. Always close the connection
        manager.close()
        
    print("--- Example script finished ---")
