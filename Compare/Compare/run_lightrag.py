import asyncio
import pandas as pd
import numpy as np
import os
from lightrag import LightRAG, QueryParam
from config_env import call_gemini, openai_client
from rag_metrics import MetricsTracker

# Wrapper cho LightRAG
async def gemini_wrapper(prompt, system_prompt=None, history_messages=[], **kwargs):
    return call_gemini(prompt, system_prompt)

async def embedding_wrapper(texts: list[str]) -> np.ndarray:
    res = openai_client.embeddings.create(input=texts, model="text-embedding-3-small")
    return np.array([d.embedding for d in res.data])

async def main():
    print("🚀 Khởi động LightRAG...")
    # Setup
    rag = LightRAG(
        working_dir="./lightrag_index",
        llm_model_func=gemini_wrapper,
        embedding_func=LightRAG.Embedding(func=embedding_wrapper, batch_size=12)
    )

    # Load Data & Index
    data_file = "viquad_completed.xlsx - Sheet1.csv"
    if not os.path.exists("./lightrag_index/kv_store"):
        print("⏳ Đang Indexing dữ liệu (Lần đầu sẽ lâu)...")
        df = pd.read_csv(data_file)
        contexts = df['context'].astype(str).unique().tolist()
        for txt in contexts:
            await rag.ainsert(txt)
    
    # Test Question (Hop 4)
    df = pd.read_csv(data_file)
    test_row = df[df['hop'] == 4].iloc[0]
    query = test_row['question']
    
    print(f"❓ Câu hỏi: {query}")
    
    tracker = MetricsTracker("LightRAG")
    tracker.start()
    # LightRAG đo generation bên trong, ở đây ta đo E2E
    result = await rag.aquery(query, param=QueryParam(mode="hybrid"))
    tracker.stop()
    
    print(f"💡 Đáp án: {result}")
    tracker.print_report()

if __name__ == "__main__":
    asyncio.run(main())