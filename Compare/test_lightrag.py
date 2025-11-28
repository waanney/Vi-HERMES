import os
import time
import asyncio
import pandas as pd
import numpy as np
import google.generativeai as genai
from openai import OpenAI
from dotenv import load_dotenv
from lightrag import LightRAG, QueryParam
from dataclasses import dataclass

# --- 1. SETUP & METRICS ---
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@dataclass
class Metrics:
    input_tokens: int = 0
    output_tokens: int = 0
    total_time: float = 0.0
    generation_time: float = 0.0

tracker = Metrics()

# --- 2. WRAPPER FUNCTIONS ---
async def gemini_complete(prompt, system_prompt=None, history_messages=[], **kwargs) -> str:
    start = time.perf_counter()
    model = genai.GenerativeModel('gemini-1.5-flash')
    full_prompt = (f"System: {system_prompt}\n" if system_prompt else "") + f"User: {prompt}"
    try:
        response = await model.generate_content_async(full_prompt)
        dur = time.perf_counter() - start
        if response.usage_metadata:
            tracker.input_tokens += response.usage_metadata.prompt_token_count
            tracker.output_tokens += response.usage_metadata.candidates_token_count
            tracker.generation_time += dur
        return response.text
    except: return ""

async def openai_embedding(texts: list[str]) -> np.ndarray:
    resp = openai_client.embeddings.create(input=texts, model="text-embedding-3-small")
    return np.array([d.embedding for d in resp.data])

# --- 3. MAIN LOGIC (INDEX + EVAL) ---
async def main():
    DATA_FILE = "viquad_completed.xlsx - Sheet1.csv"
    INDEX_DIR = "./lightrag_index_viquad"
    
    print(f"--- 🚀 LIGHTRAG SETUP ---")
    rag = LightRAG(
        working_dir=INDEX_DIR,
        llm_model_func=gemini_complete,
        embedding_func=LightRAG.Embedding(func=openai_embedding, batch_size=12)
    )

    # --- GIAI ĐOẠN 1: INDEXING (BUILD) ---
    # Kiểm tra nếu thư mục chứa dữ liệu chưa có file JSON thì mới Index
    if not os.path.exists(os.path.join(INDEX_DIR, "kv_store")):
        print(f"⏳ Chưa có Index. Đang đọc file {DATA_FILE} để Build...")
        df = pd.read_csv(DATA_FILE)
        contexts = df['context'].astype(str).unique().tolist()
        
        # Insert dữ liệu (Tốn phí API & Thời gian)
        for i, ctx in enumerate(contexts):
            print(f"   + Indexing doc {i+1}/{len(contexts)}...")
            await rag.ainsert(ctx)
        print("✅ Indexing hoàn tất!")
    else:
        print("✅ Đã tìm thấy Index cũ. Bỏ qua bước Build.")

    # --- GIAI ĐOẠN 2: EVALUATION (TEST) ---
    print(f"\n--- 🧪 EVALUATION (HOP 4) ---")
    df = pd.read_csv(DATA_FILE)
    # Lấy câu hỏi khó (Hop = 4)
    test_row = df[df['hop'] == 4].iloc[0]
    query = test_row['question']
    
    print(f"❓ Câu hỏi: {query}")
    
    start_total = time.perf_counter()
    # Query model
    result = await rag.aquery(query, param=QueryParam(mode="hybrid"))
    end_total = time.perf_counter()
    
    tracker.total_time = end_total - start_total
    
    # --- REPORT ---
    print(f"\n{'='*30}")
    print(f"💡 Đáp án Model: {result}")
    print(f"📝 Đáp án Gốc:  {test_row['answers']}")
    print(f"{'='*30}")
    print(f"📊 METRICS:")
    print(f"   - Tổng thời gian: {tracker.total_time:.4f}s")
    print(f"   - Thời gian LLM suy nghĩ: {tracker.generation_time:.4f}s")
    print(f"   - Thời gian Retrieval: {max(0, tracker.total_time - tracker.generation_time):.4f}s")
    print(f"   - Token Input: {tracker.input_tokens} | Output: {tracker.output_tokens}")

if __name__ == "__main__":
    asyncio.run(main())