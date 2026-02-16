import argparse
import re
import sys
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()

from langchain_community.document_loaders import TextLoader
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter

from hanuman.settings import settings


def clean_text(text: str) -> str:
    text = re.sub(r"\n+", "\n", text)
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE)
    return text.strip()


def get_vector_store() -> PGVector:
    embeddings = HuggingFaceEmbeddings(model=settings.embedding.model)

    return PGVector(
        embeddings=embeddings,
        connection=settings.database.connection_string,
        collection_name="documents",
    )


def index_mode(input_file: str) -> None:
    path = Path(input_file)
    if not path.exists():
        print(f"Error: File '{input_file}' not found")
        sys.exit(1)

    print(f"Loading file: {input_file}")
    loader = TextLoader(str(path), encoding="utf-8")
    documents = loader.load()

    for doc in documents:
        doc.page_content = clean_text(doc.page_content)

    print(f"Loaded {len(documents)} document(s)")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        length_function=len,
    )

    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")

    vector_store = get_vector_store()
    vector_store.add_documents(chunks)
    print("Indexed successfully!")


def search_mode(query: str) -> None:
    if not settings.groq_api_key:
        print("Error: GROQ_API_KEY is not set")
        sys.exit(1)

    print(f"Searching for: {query}")

    vector_store = get_vector_store()
    docs = vector_store.similarity_search(query, k=settings.top_k)

    if not docs:
        print("No results found")
        return

    context = "\n\n".join(doc.page_content for doc in docs)

    prompt = f"""Based on the following context, answer the question.

Context:
{context}

Question: {query}

Answer:"""

    llm = ChatGroq(
        model=settings.chat.model,
        groq_api_key=settings.groq_api_key.get_secret_value() if settings.groq_api_key else None,  # type: ignore[arg-type]
        temperature=settings.chat.temperature,
        max_tokens=settings.chat.max_tokens,
    )

    response = llm.invoke(prompt)
    print("\n--- Response ---")
    print(response.content)


def main() -> None:
    parser = argparse.ArgumentParser(description="Hanuman - 2-Step RAG Application")
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index", help="Index a text file")
    index_parser.add_argument("input", help="Input text file path")

    search_parser = subparsers.add_parser("search", help="Search using RAG")
    search_parser.add_argument("query", help="Search query")

    args = parser.parse_args()

    if args.command == "index":
        index_mode(args.input)
    elif args.command == "search":
        search_mode(args.query)


if __name__ == "__main__":
    main()
