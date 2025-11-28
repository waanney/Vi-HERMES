import os
import time
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
import google.generativeai as genai
from dotenv import load_dotenv

# --- 1. SETUP ---
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Metrics
metrics = {"in_tok": 0, "out_tok": 0, "gen_time": 0, "start_time": 0}

# --- 2. COMPONENTS ---
class IRCoTSystem:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.chroma_path = "./ircot_chroma_db"
        
        # Setup ChromaDB & OpenAI Embedding
        self.client = chromadb.PersistentClient(path=self.chroma_path)
        self.openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"), model_name="text-embedding-3-small"
        )
        # Tự động Build hoặc Load
        self._setup_collection()
        
    def _setup_collection(self):
        try:
            # Thử lấy collection cũ
            self.collection = self.client.get_collection("viquad_db", embedding_function=self.openai_ef)
            print("✅ Đã load Vector DB từ ổ cứng.")
        except:
            # Nếu chưa có -> Build mới
            print("⏳ Đang Build Vector DB từ file CSV...")
            self.collection = self.client.create_collection("viquad_db", embedding_function=self.openai_ef)
            df = pd.read_csv(self.csv_path)
            
            # Add data theo batch
            batch_size = 50
            for i in range(0, len(df), batch_size):
                batch = df.iloc[i:i+batch_size]
                self.collection.add(
                    ids=[str(x) for x in batch['id']],
                    documents=batch['context'].astype(str).tolist()
                )
            print(f"✅ Đã Index xong {len(df)} dòng.")

    def retrieve(self, query):
        results = self.collection.query(query_texts=[query], n_results=1)
        return results['documents'][0][0] if results['documents'] else ""

    def reason_llm(self, prompt):
        t0 = time.perf_counter()
        model = genai.GenerativeModel('gemini-1.5-flash')
        res = model.generate_content(prompt)
        dur = time.perf_counter() - t0
        
        if res.usage_metadata:
            metrics["in_tok"] += res.usage_metadata.prompt_token_count
            metrics["out_tok"] += res.usage_metadata.candidates_token_count
            metrics["gen_time"] += dur
        return res.text

# --- 3. MAIN RUN ---
def main():
    system = IRCoTSystem("viquad_completed.xlsx - Sheet1.csv")
    
    # Lấy câu hỏi Hop 4
    df = pd.read_csv("viquad_completed.xlsx - Sheet1.csv")
    test_case = df[df['hop'] == 4].iloc[0]
    query = test_case['question']
    max_steps = int(test_case['hop'])

    print(f"\n❓ IRCoT chạy câu hỏi (Max Hop {max_steps}): {query}")
    
    metrics["start_time"] = time.perf_counter()
    context = ""
    
    # Vòng lặp IRCoT
    for step in range(max_steps):
        # Retrieve
        evidence = system.retrieve(query) # Có thể cải tiến query tại đây
        context += f"\n[Step {step+1} Evidence]: {evidence[:300]}..."
        
        # Reason
        prompt = f"Question: {query}\nEvidence: {context}\nSuy luận tiếp (Ghi 'ANSWER:' nếu xong):"
        thought = system.reason_llm(prompt)
        print(f"👉 Step {step+1}: {thought.strip()[:100]}...")
        
        if "ANSWER:" in thought: break

    total_time = time.perf_counter() - metrics["start_time"]
    
    print(f"\n{'='*30}\n📊 KẾT QUẢ ĐO LƯỜNG:")
    print(f"   - Tổng thời gian: {total_time:.4f}s")
    print(f"   - Generation Time: {metrics['gen_time']:.4f}s")
    print(f"   - Retrieval Time: {total_time - metrics['gen_time']:.4f}s")
    print(f"   - Tokens: Input {metrics['in_tok']} / Output {metrics['out_tok']}")

if __name__ == "__main__":
    main()