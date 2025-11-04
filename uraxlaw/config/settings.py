from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Neo4j
    neo4j_uri: str = Field(default="bolt://localhost:7687")
    neo4j_user: str = Field(default="neo4j")
    neo4j_password: str = Field(default="password123")

    # Milvus
    milvus_host: str = Field(default="localhost")
    milvus_port: int = Field(default=19530)
    milvus_collection: str = Field(default="uraxlaw_articles")

    # LLM
    llm_model: str = Field(default="gpt-4o")

    model_config = {
        "env_file": ".env",
    }


def get_settings() -> Settings:
    return Settings()
