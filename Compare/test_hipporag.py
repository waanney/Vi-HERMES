import os
import json
import time
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv

# --- 1. SETUP ---
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

metrics = {"in_tok": 0, "out_tok": 0, "gen_time": 0}

# --- 2. DATA PREPARATION (INDEXING PHASE 1) ---
def prepare_data(csv_path):
    output_json = "data/viquad_corpus.json"
    if not os.path.exists(output_json):
        print("⏳ Đang tạo file JSON cho HippoRAG từ CSV...")
        df = pd.read_csv(csv_path)
        corpus = []
        for _, row in df.iterrows():
            corpus.append({
                "id": str(row['id']),
                "title": str(row['title']),
                "text": str(row['context'])
            })
        os.makedirs("data", exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(corpus, f, ensure_ascii=False, indent=2)
        print("✅ Đã tạo corpus.json")
    return output_json

# --- 3. CUSTOM LLM WRAPPER ---
class GeminiWrapper:
    def __init__(self, model_name='gemini-1.5-flash'):
        self.model = genai.GenerativeModel(model_name)
    
    def generate(self, prompt, **kwargs):
        t0 = time.perf_counter()
        try:
            res = self.model.generate_content(prompt)
            dur = time.perf_counter() - t0
            if res.usage_metadata:
                metrics["in_tok"] += res.usage_metadata.prompt_token_count
                metrics["out_tok"] += res.usage_metadata.candidates_token_count
                metrics["gen_time"] += dur
            return res.text
        except: return ""

# --- 4. MAIN ---
def main():
    CSV_FILE = "viquad_completed.xlsx - Sheet1.csv"
    prepare_data(CSV_FILE)
    
    # Import HippoRAG (Chỉ chạy được khi file này nằm trong repo HippoRAG)
    try:
        from src.hipporag import HippoRAG
    except ImportError:
        print("❌ LỖI: File này phải nằm trong thư mục gốc của repo HippoRAG!")
        return

    # Lấy câu hỏi test
    df = pd.read_csv(CSV_FILE)
    test_row = df[df['hop'] == 4].iloc[0]
    query = test_row['question']
    
    print(f"🚀 HippoRAG đang chạy câu hỏi: {query}")
    
    # Init Engine
    # Lưu ý: Lần đầu chạy dòng này, HippoRAG sẽ tự build Graph Index (rất lâu)
    rag = HippoRAG(
        corpus_path="data/viquad_corpus.json", 
        llm=GeminiWrapper()
    )
    
    start_time = time.perf_counter()
    answer = rag.predict(query) # Hoặc hàm tương đương tuỳ phiên bản
    total_time = time.perf_counter() - start_time
    
    print(f"\n💡 Kết quả: {answer}")
    print(f"\n📊 METRICS:")
    print(f"   - Tổng thời gian: {total_time:.4f}s")
    print(f"   - Generation Time: {metrics['gen_time']:.4f}s")
    print(f"   - Retrieval Time: {total_time - metrics['gen_time']:.4f}s")

if __name__ == "__main__":
    main()