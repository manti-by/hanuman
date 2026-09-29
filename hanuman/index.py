import asyncio
import re
import sys
from pathlib import Path

import yaml
from langchain_core.documents import Document

from hanuman.services.store import get_vector_store_context
from hanuman.services.tui import print_message
from hanuman.services.utils import clean_text
from hanuman.settings import settings


FRONT_MATTER_RE = re.compile(r"\A---[ \t]*\r?\n(.*?)\r?\n---[ \t]*(?:\r?\n|\Z)", re.DOTALL)
TABLE_DELIMITER_RE = re.compile(r"^\|[\s:|-]+\|[ \t]*$", re.MULTILINE)


def strip_markdown(text: str) -> str:
    text = TABLE_DELIMITER_RE.sub("", text)
    text = re.sub(r"^#{1,6}[ \t]+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^[ \t]*\|[ \t]*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\|[ \t]*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*\|\s*", " ", text)
    text = re.sub(r"\*\*|__|[*`]", "", text)
    return clean_text(text)


def parse_markdown(path: Path) -> Document:
    text = path.read_text(encoding="utf-8")
    metadata: dict[str, object] = {}
    match = FRONT_MATTER_RE.match(text)
    if match:
        front_matter = yaml.safe_load(match.group(1))
        if isinstance(front_matter, dict):
            metadata.update(front_matter)
        text = text[match.end() :]
    metadata["file"] = path.name
    return Document(page_content=strip_markdown(text), metadata=metadata)


def load_documents(source: Path) -> list[Document]:
    files = sorted(source.rglob("*.md")) if source.is_dir() else [source]
    if not files:
        print_message(f"Error: No markdown files found in '{source}'", style="error")
        sys.exit(1)

    print_message(f"Loading {len(files)} file(s) from {source}", style="heading")
    documents = [document for document in map(parse_markdown, files) if document.page_content]
    if not documents:
        print_message(f"Error: No content found in '{source}'", style="error")
        sys.exit(1)

    prefix = settings.embedding.document_prefix
    for document in documents:
        document.page_content = f"{prefix}{document.page_content}"

    print_message(f"Loaded {len(documents)} chunk(s)", style="result")
    return documents


async def index(input_path: str | None = None) -> None:
    source = Path(input_path) if input_path else settings.chunks_dir
    if not source.exists():
        print_message(f"Error: Path '{source}' not found", style="error")
        sys.exit(1)

    documents = load_documents(source)

    print_message("Indexing documents", style="heading")
    async with get_vector_store_context() as vector_store:
        await asyncio.to_thread(vector_store.add_documents, documents)
    print_message("Index is finished successfully", style="result")
