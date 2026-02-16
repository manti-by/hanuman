# Hanuman - 2-Step RAG Application

A classic Retrieval-Augmented Generation (RAG) application built with LangChain.

## Features

- **Index Mode**: Load text files, clean, chunk, embed, and store in PostgreSQL with pgvector
- **Search Mode**: Retrieve relevant context from vector store and generate answers using Groq LLM
- **Docker Support**: Complete docker-compose setup with PostgreSQL + pgvector

## Tech Stack

- **Framework**: LangChain
- **Chat Model**: Groq (llama-3.1-8b-instant)
- **Vector Store**: PostgreSQL with pgvector extension
- **Embeddings**: HuggingFace (multilingual-e5-large)

## Quick Start

### Using Docker

1. Copy the environment file and add your Groq API key:
   ```bash
   cp .env.example .env
   # Edit .env and add your GROQ_API_KEY
   ```

2. Start the services:
   ```bash
   docker-compose up -d
   ```

3. Access the app container:
   ```bash
   docker-compose exec app bash
   ```

4. Index a text file:
   ```bash
   python main.py index input/document.txt
   ```

5. Search:
   ```bash
   python main.py search "your search query"
   ```

### Local Development

1. Install dependencies:
   ```bash
   uv sync --all-extras --dev
   ```

2. Start PostgreSQL with pgvector (or use Docker):
   ```bash
   docker run -d -p 5432:5432 -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=hanuman pgvector/pgvector:pg16
   ```

3. Set environment variables:
   ```bash
   export GROQ_API_KEY=your_api_key
   ```

4. Run commands:
   ```bash
   uv run main.py index input/example.txt
   uv run main.py search "How does temperature affect egg formation?"
   ```

## Configuration

Configuration can be set via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | - | Groq API key |
| `DATABASE__HOST` | localhost | Database host |
| `DATABASE__PORT` | 5432 | Database port |
| `DATABASE__USER` | postgres | Database user |
| `DATABASE__PASSWORD` | postgres | Database password |
| `DATABASE__DATABASE` | hanuman | Database name |
| `CHUNK_SIZE` | 1000 | Text chunk size |
| `CHUNK_OVERLAP` | 200 | Chunk overlap |
| `TOP_K` | 4 | Number of similar documents to retrieve |

## Testing

```bash
make test
# or
uv run pytest tests/
```
