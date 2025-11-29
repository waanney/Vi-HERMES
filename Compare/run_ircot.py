import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from config_env import call_gemini, OPENAI_KEY
from rag_metrics import MetricsTracker

class IRCoTSystem:
    def __init__(self, csv_path):
        self.chroma_client = chromadb.PersistentClient(path="./ircot_db")
        self.ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=OPENAI_KEY, model_name="text-embedding-3-small"
        )
        self.collection = self.chroma_client.get_or_create_collection("viquad", embedding_function=self.ef)
        
        # Index nếu trống
        if self.collection.count() == 0:
            print("⏳ Đang Indexing IRCoT...")
            df = pd.read_csv(csv_path)
            batch_size = 50
            for i in range(0, len(df), batch_size):
                batch = df.iloc[i:i+batch_size]
                self.collection.add(
                    ids=[str(x) for x in batch['id']],
                    documents=batch['context'].astype(str).tolist()
                )

    def retrieve(self, query):
        res = self.collection.query(query_texts=[query], n_results=1)
        return res['documents'][0][0] if res['documents'] else ""

def main():
    print("🚀 Khởi động IRCoT...")
    system = IRCoTSystem("viquad_completed.xlsx - Sheet1.csv")
    tracker = MetricsTracker("IRCoT")
    
    # Lấy câu hỏi
    df = pd.read_csv("viquad_completed.xlsx - Sheet1.csv")
    test_case = df[df['hop'] == 4].iloc[0]
    query = test_case['question']
    max_steps = int(test_case['hop'])
    
    print(f"❓ Câu hỏi (Max Hop {max_steps}): {query}")
    
    tracker.start()
    context = ""
    
    for step in range(max_steps):
        # 1. Retrieve
        evidence = system.retrieve(query) # Trong thực tế query có thể thay đổi theo thought
        context += f"\n[Step {step+1}]: {evidence[:300]}..."
        
        # 2. Reason (Gemini)
        tracker.start_gen()
        prompt = f"Question: {query}\nEvidence so far: {context}\nSuy luận tiếp (Ghi 'ANSWER:' nếu xong):"
        thought = call_gemini(prompt)
        tracker.end_gen()
        
        print(f"👉 Step {step+1}: {thought.strip()[:100]}...")
        if "ANSWER:" in thought:
            break
            
    tracker.stop()
    tracker.print_report()

if __name__ == "__main__":
    main()