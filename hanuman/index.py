import sys
from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from hanuman.services.store import get_vector_store
from hanuman.services.utils import clean_text
from hanuman.settings import settings


async def index(input_file: str) -> None:
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

    vector_store = await get_vector_store()
    vector_store.add_documents(chunks)
    print("Indexed successfully!")
