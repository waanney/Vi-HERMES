import os
import json
import pandas as pd
from config_env import call_gemini
from rag_metrics import MetricsTracker
import time

# Wrapper
class GeminiHippoWrapper:
    def generate(self, prompt, **kwargs):
        return call_gemini(prompt)

def prepare_hipporag_data(csv_path):
    output = "data/viquad_corpus.json"
    if not os.path.exists(output):
        print("⚙️ Đang tạo JSON cho HippoRAG...")
        df = pd.read_csv(csv_path)
        corpus = []
        for _, row in df.iterrows():
            corpus.append({"id": str(row['id']), "title": str(row['title']), "text": str(row['context'])})
        os.makedirs("data", exist_ok=True)
        with open(output, "w", encoding="utf-8") as f:
            json.dump(corpus, f, ensure_ascii=False)
    return output

def main():
    print("🚀 Khởi động HippoRAG 2...")
    try:
        from src.hipporag import HippoRAG # Import thư viện nội bộ
    except ImportError:
        print("❌ LỖI: File này phải nằm trong folder gốc của repo HippoRAG!")
        return

    csv_path = "viquad_completed.xlsx - Sheet1.csv"
    prepare_hipporag_data(csv_path)
    
    # Init
    rag = HippoRAG(corpus_path="data/viquad_corpus.json", llm=GeminiHippoWrapper())
    
    # Test Query
    df = pd.read_csv(csv_path)
    query = df[df['hop'] == 4].iloc[0]['question']
    print(f"❓ Câu hỏi: {query}")
    
    tracker = MetricsTracker("HippoRAG")
    tracker.start()
    
    # Mockup timing cho generation vì HippoRAG gọi ẩn bên trong
    tracker.start_gen() 
    answer = rag.predict(query)
    tracker.end_gen()
    
    tracker.stop()
    print(f"💡 Đáp án: {answer}")
    tracker.print_report()

if __name__ == "__main__":
    main()