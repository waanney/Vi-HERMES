import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from config_env import call_gemini, get_embedding
from rag_metrics import MetricsTracker

class RaptorRAG:
    def __init__(self, contexts):
        self.contexts = contexts
        self.embeddings = []
        self.tree_nodes = [] # Chứa chunks gốc + summary nodes
        self.node_embeddings = []

    def build(self):
        print("1. Embedding dữ liệu gốc...")
        self.embeddings = np.array([get_embedding(txt) for txt in self.contexts])
        self.tree_nodes = self.contexts.copy()
        
        print("2. Phân cụm & Tóm tắt (RAPTOR Layer)...")
        # Chia thành 5 cụm (hoặc dynamic)
        n_clusters = 5
        gmm = GaussianMixture(n_components=n_clusters, random_state=42)
        gmm.fit(self.embeddings)
        labels = gmm.predict(self.embeddings)
        
        for i in range(n_clusters):
            cluster_docs = [self.contexts[idx] for idx in np.where(labels == i)[0]]
            # Gọi Gemini tóm tắt cụm
            combined_text = "\n".join(cluster_docs[:5]) # Lấy 5 docs đại diện để tóm tắt
            summary = call_gemini(f"Tóm tắt ngắn gọn nội dung chung của các đoạn văn sau:\n{combined_text}")
            self.tree_nodes.append(f"[SUMMARY CLUSTER {i}]: {summary}")
            print(f"   - Đã tạo Summary cụm {i}")

        # Embed lại toàn bộ nodes (gốc + summary)
        self.node_embeddings = np.array([get_embedding(txt) for txt in self.tree_nodes])

    def query(self, q):
        q_vec = get_embedding(q)
        # Tìm kiếm trên cây đã gộp (Collapsed Tree Retrieval)
        scores = np.dot(self.node_embeddings, q_vec)
        top_idx = np.argsort(scores)[::-1][:3]
        return "\n---\n".join([self.tree_nodes[i] for i in top_idx])

def main():
    print("🚀 Khởi động RAPTOR...")
    df = pd.read_csv("viquad_completed.xlsx - Sheet1.csv")
    # Lấy 50 contexts đầu để demo cho nhanh (chạy hết sẽ lâu)
    contexts = df['context'].astype(str).unique().tolist()[:50]
    
    raptor = RaptorRAG(contexts)
    raptor.build()
    
    # Test
    query = df[df['hop'] == 4].iloc[0]['question']
    print(f"❓ Câu hỏi: {query}")
    
    tracker = MetricsTracker("RAPTOR")
    tracker.start()
    
    # 1. Retrieve
    context = raptor.query(query)
    
    # 2. Answer
    tracker.start_gen()
    answer = call_gemini(f"Context:\n{context}\nQuestion: {query}\nAnswer:")
    tracker.end_gen()
    
    tracker.stop()
    print(f"💡 Đáp án: {answer}")
    tracker.print_report()

if __name__ == "__main__":
    main()