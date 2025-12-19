# Vi-HERMES

Vi-HERMES is a Retrieval-Augmented Generation (RAG) system for Vietnamese legal documents. It combines vector search (knowledge base) and graph traversal (knowledge graph) to provide accurate answers to legal queries.

## Overview

The system processes Vietnamese legal documents and makes them searchable through:
- Vector similarity search using Milvus
- Graph-based relationship traversal using Neo4j
- Hybrid retrieval combining both methods
- LLM-powered answer generation

## Requirements

- Python >= 3.10
- Docker and Docker Compose
- OpenAI API key (or Gemini API key for alternative LLM)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd UraxLawv1
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e .
```

Alternatively, using uv:
```bash
uv pip install -e .
```

3. Create a `.env` file in the project root with the following variables:
```
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password123
MILVUS_HOST=localhost
MILVUS_PORT=19530
MILVUS_COLLECTION=vihermes_articles
OPENAI_API_KEY=your_openai_api_key_here
LLM_MODEL=gpt-4o
```

## Running the Infrastructure

Start the required services (Neo4j, Milvus, etcd, MinIO) using Docker Compose:

```bash
cd docker
docker network create milvus  # Only needed once
docker-compose up -d
```

This will start:
- Neo4j on port 7687 (Bolt) and 7474 (Web UI)
- Milvus on port 19530
- Milvus Attu (web UI) on port 8000
- etcd and MinIO (required by Milvus)

To stop the services:
```bash
cd docker
docker-compose down
```

## Project Structure

### Root Directory

- `pyproject.toml` - Project configuration and dependencies
- `flake.nix` - Nix development environment configuration
- `uv.lock` - Dependency lock file
- `.env` - Environment variables (create this file)

### vihermes/

Main package containing the core modules:

#### vihermes/Agents/
- `engine.py` - GraphRAGEngine for query processing and answer generation
- `models.py` - Data models for agents
- `prompt.py` - Prompt building utilities
- `relations.py` - Relation extraction logic
- `segmentation.py` - Document segmentation

#### vihermes/config/
- `settings.py` - Configuration settings loaded from environment variables

#### vihermes/lawgraph/
- `models.py` - Graph data models (Node, Edge)
- `neo4j_client.py` - Neo4j client wrapper
- `traversal.py` - Graph traversal algorithms

#### vihermes/lawrag/
- `hybrid.py` - Hybrid retriever combining vector and graph search
- `milvus_client.py` - Milvus client wrapper
- `models.py` - RAG data models

#### vihermes/preprocess/
- `agent_chunker.py` - Agent-based document chunking
- `ingestion.py` - Document ingestion pipeline
- `milvus_chunker.py` - Chunking for Milvus storage
- `neo4j_chunker.py` - Chunking for Neo4j graph
- `parser.py` - Document parsing utilities
- `pipeline.py` - Complete preprocessing pipeline

### config/
Configuration files directory (if needed)

### data/
Data directory containing:
- Legal documents in text format
- CSV datasets (ViQuAD, multihop QA datasets)
- Prompts and other data files

### docker/
Docker Compose configuration and related files:
- `docker-compose.yml` - Service definitions for Neo4j, Milvus, etcd, MinIO
- `neo4j/` - Neo4j data and configuration volumes
- `volumes/` - Persistent volumes for Milvus data

### examples/
Example scripts demonstrating various features:
- `query_example.py` - Basic query example
- `query_rag.py` - Chainlit-based interactive query interface
- `ingest_viquad_full_levels.py` - Ingest ViQuAD dataset into Milvus and Neo4j
- `ingest_viquad_to_milvus.py` - Ingest data to Milvus only
- `test_preprocess.py` - Test preprocessing pipeline
- `test_neo4j_chunker.py` - Test Neo4j chunking
- `test_parse_documents.py` - Test document parsing
- `test_parser_to_milvus.py` - Test Milvus ingestion
- `test_parser_to_neo4j.py` - Test Neo4j ingestion
- `test_three_agents.py` - Test agent system
- `evaluate_rag_on_viquad.py` - Evaluate RAG performance on ViQuAD dataset
- `count_graph_stats.py` - Count graph statistics
- `run_200_last_rows.py` - Process last 200 rows from dataset
- `clear_old_data.py` - Clear old data from stores

## Usage

### Basic Query Example

Run a simple query using the command-line interface:

```bash
python examples/query_example.py
```

### Interactive Query Interface (Chainlit)

Start the interactive web interface:

```bash
cd examples
chainlit run query_rag.py
```

Then open your browser to the URL shown (typically http://localhost:8000).

### Ingest Documents

Ingest the ViQuAD dataset into both Milvus and Neo4j:

```bash
python examples/ingest_viquad_full_levels.py \
    --input data/viquad_full_levels_cleaned.csv \
    --batch-size 100 \
    --recreate
```

Options:
- `--input`: Path to CSV file
- `--batch-size`: Number of records per batch (default: 100)
- `--max-rows`: Maximum rows to process (optional, for testing)
- `--start-row`: Starting row index (for resuming)
- `--recreate`: Recreate collection (deletes existing data)

### Test Components

Test the preprocessing pipeline:
```bash
python examples/test_preprocess.py
```

Test Neo4j chunking:
```bash
python examples/test_neo4j_chunker.py
```

Test document parsing:
```bash
python examples/test_parse_documents.py
```

### Evaluate Performance

Evaluate RAG performance on the ViQuAD dataset:
```bash
python examples/evaluate_rag_on_viquad.py
```

## Configuration

Configuration is managed through environment variables (via `.env` file) and the `Settings` class in `vihermes/config/settings.py`. The following settings are available:

- Neo4j connection settings
- Milvus connection settings
- LLM model and API key
- Collection names

## Architecture

The system follows a hybrid RAG architecture:

1. **Preprocessing**: Documents are parsed, chunked, and metadata is extracted
2. **Storage**: 
   - Vector embeddings stored in Milvus for similarity search
   - Graph nodes and edges stored in Neo4j for relationship traversal
3. **Retrieval**: Hybrid retriever combines results from both stores
4. **Generation**: LLM generates answers based on retrieved context

## Dependencies

Key dependencies include:
- chainlit - Interactive web UI
- neo4j - Graph database driver
- openai - OpenAI API client
- pydantic-ai - Pydantic-based AI framework
- sentence-transformers - Embedding models
- pymilvus - Milvus vector database client
- python-docx - Word document parsing
- google-generativeai - Gemini API support (optional)

## License

MIT License

