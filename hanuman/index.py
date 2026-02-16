import sys
from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from hanuman.services.store import get_vector_store
from hanuman.services.tui import print_message
from hanuman.services.utils import clean_text
from hanuman.settings import settings


async def index(input_file: str) -> None:
    print_message(f"Loading file: {input_file}", style="heading")
    path = Path(input_file)
    if not path.exists():
        print_message(f"Error: File '{input_file}' not found", style="error")
        sys.exit(1)

    loader = TextLoader(str(path), encoding="utf-8")
    documents = loader.load()
    for doc in documents:
        doc.page_content = clean_text(doc.page_content)
    print_message(f"Loaded {len(documents)} document(s)", style="result")

    print_message("Splitting file into chunks", style="heading")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    print_message(f"Created {len(chunks)} chunks", style="result")

    print_message("Indexing documents", style="heading")
    vector_store = await get_vector_store()
    vector_store.add_documents(chunks)
    print_message("Index is finished successfully", style="result")
