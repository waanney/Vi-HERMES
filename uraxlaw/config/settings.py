from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # Neo4j settings
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password123"

    # Milvus settings
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_collection: str = "legal_document_collection"
    milvus_uri: Optional[str] = None

    # LLM settings (Thêm dòng này để pydantic nhận diện key)
    openai_api_key: Optional[str] = None
    llm_model: str = "gpt-4o"
    embedding_model: str = "text-embedding-3-small"

    # Cấu hình: extra="ignore" để không báo lỗi khi file .env có nhiều biến lạ
    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8",
        extra="ignore" 
    )

def get_settings() -> Settings:
    return Settings()
