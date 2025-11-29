import os
import google.generativeai as genai
from openai import OpenAI
from dotenv import load_dotenv

# Load key từ .env
load_dotenv()
GEMINI_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

if not GEMINI_KEY or not OPENAI_KEY:
    raise ValueError("❌ CHƯA CÓ API KEY TRONG FILE .ENV")

# 1. Setup Gemini
genai.configure(api_key=GEMINI_KEY)
# Dùng bản Flash cho nhanh và rẻ
gemini_model = genai.GenerativeModel('gemini-2.0-flash')

# 2. Setup OpenAI Embedding
openai_client = OpenAI(api_key=OPENAI_KEY)

# --- HÀM DÙNG CHUNG ---
def call_gemini(prompt, system_instruction=None):
    """Hàm gọi Gemini trả về text"""
    try:
        final_prompt = prompt
        if system_instruction:
            final_prompt = f"System: {system_instruction}\nUser: {prompt}"
            
        response = gemini_model.generate_content(final_prompt)
        return response.text if response.text else ""
    except Exception as e:
        print(f"⚠️ Gemini Error: {e}")
        return ""

def get_embedding(text):
    """Hàm gọi OpenAI Embedding trả về vector"""
    text = str(text).replace("\n", " ")
    return openai_client.embeddings.create(input=[text], model="text-embedding-3-small").data[0].embedding