import asyncio
import re
from collections import Counter
from pathlib import Path

import pymupdf4llm
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from hanuman.services.tui import print_message
from hanuman.settings import settings


HEADING_RE = re.compile(r"^(?P<hashes>#{1,6})\s+(?P<text>.+?)\s*$")
SECTION_RE = re.compile(r"^(?P<number>\d+(?:\.\d+)*)\.?(?=\s|$)")
CAPTION_RE = re.compile(r"^(?:Таблица|Рисунок|Рис\.)\b")
MARKUP_RE = re.compile(r"<sup>.*?</sup>|\*\*|__|`")
PAGE_NUMBER_RE = re.compile(r"^\d{1,4}$")
RUNNING_HEADER_MIN_REPEATS = 5
RUNNING_HEADER_MAX_LENGTH = 80
MAX_HEADING_LEVEL = 6

HEADERS_TO_SPLIT_ON: list[tuple[str, str]] = [
    ("#" * level, f"Header {level}") for level in range(1, MAX_HEADING_LEVEL + 1)
]


def _strip_markup(text: str) -> str:
    return re.sub(r"\s+", " ", MARKUP_RE.sub("", text)).strip()


def _heading_level(heading: str, fallback: int) -> int:
    match = SECTION_RE.match(heading)
    if match:
        return min(match["number"].count(".") + 1, MAX_HEADING_LEVEL)
    return min(fallback, MAX_HEADING_LEVEL)


def _join_wrapped(head: str, tail: str) -> str:
    if head.endswith("-") and tail[:1].islower():
        return f"{head[:-1]}{tail}"
    return f"{head} {tail}"


def strip_page_furniture(markdown: str) -> str:
    lines = markdown.split("\n")
    counts = Counter(line.strip() for line in lines)
    running_headers = {
        line
        for line, count in counts.items()
        if count >= RUNNING_HEADER_MIN_REPEATS
        and len(line) <= RUNNING_HEADER_MAX_LENGTH
        and line.startswith("**")
        and line.endswith("**")
    }
    if not running_headers:
        return markdown

    return "\n".join(
        line for line in lines if line.strip() not in running_headers and not PAGE_NUMBER_RE.match(line.strip())
    )


def normalize_headings(markdown: str) -> str:
    blocks: list[str] = []
    last_heading: int | None = None

    for line in strip_page_furniture(markdown).split("\n"):
        match = HEADING_RE.match(line)
        if not match:
            blocks.append(line)
            continue

        heading = _strip_markup(match["text"])
        if not heading:
            continue

        numbered = bool(SECTION_RE.match(heading))
        if last_heading is not None and not numbered:
            blocks[last_heading] = _join_wrapped(blocks[last_heading], heading)
            continue

        if not numbered and CAPTION_RE.match(heading):
            blocks.append(f"**{heading}**")
            last_heading = None
            continue

        blocks.append(f"{'#' * _heading_level(heading, len(match['hashes']))} {heading}")
        last_heading = len(blocks) - 1

    return "\n".join(blocks)


def split_markdown(markdown: str, source: str) -> list[Document]:
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=HEADERS_TO_SPLIT_ON)
    bound_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        length_function=len,
    )

    documents: list[Document] = []
    for chunk in splitter.split_text(markdown):
        keys = sorted(key for key in chunk.metadata if key.startswith("Header "))
        headers = [chunk.metadata[key] for key in keys]
        metadata = {"source": source, "headers": headers, "section": " > ".join(headers)}
        for piece in bound_splitter.split_text(chunk.page_content):
            if piece.strip():
                documents.append(Document(page_content=piece, metadata=dict(metadata)))
    return documents


def _slugify(heading: str) -> str:
    slug = re.sub(r"[^\w\s-]", "", heading, flags=re.UNICODE).strip().lower()
    return re.sub(r"[-\s]+", "-", slug)[:60].strip("-") or "chunk"


def _chunk_filename(document: Document, position: int) -> str:
    headers = document.metadata["headers"]
    for header in reversed(headers):
        match = SECTION_RE.match(header)
        if match:
            return f"{position:04d}-{match['number'].replace('.', '-')}.md"
    return f"{position:04d}-{_slugify(headers[-1] if headers else 'chunk')}.md"


def write_chunks(documents: list[Document], source: str) -> int:
    directory = settings.chunks_dir / Path(source).stem
    directory.mkdir(parents=True, exist_ok=True)
    for stale in directory.glob("*.md"):
        stale.unlink()

    for position, document in enumerate(documents, start=1):
        headers = "\n".join(f"- {header}" for header in document.metadata["headers"])
        front_matter = f"---\nsource: {source}\nheaders:\n{headers}\n---\n\n"
        (directory / _chunk_filename(document, position)).write_text(
            front_matter + document.page_content, encoding="utf-8"
        )
    return len(documents)


async def convert_pdfs_to_markdown() -> list[Path]:
    pdf_files = sorted(settings.raw_dir.glob("*.pdf"))
    if not pdf_files:
        print_message(f"No PDF files found in {settings.raw_dir}", style="error")
        return []

    settings.markdown_dir.mkdir(parents=True, exist_ok=True)
    print_message(f"Converting {len(pdf_files)} PDF file(s) to markdown", style="heading")

    converted: list[Path] = []
    for pdf_file in pdf_files:
        markdown = await asyncio.to_thread(
            pymupdf4llm.to_markdown,
            str(pdf_file),
            show_progress=False,
        )
        target = settings.markdown_dir / f"{pdf_file.stem}.md"
        target.write_text(markdown, encoding="utf-8")
        converted.append(target)
        print_message(f"{pdf_file.name} -> {target.name} ({len(markdown)} chars)", style="result")

    return converted


async def prepare() -> None:
    markdown_files = await convert_pdfs_to_markdown()
    if not markdown_files:
        return

    print_message("Splitting markdown into chunks", style="heading")
    total = 0
    for markdown_file in markdown_files:
        markdown = normalize_headings(markdown_file.read_text(encoding="utf-8"))
        total += write_chunks(split_markdown(markdown, markdown_file.stem), markdown_file.stem)
        print_message(f"{markdown_file.stem}: {total} chunk(s) so far", style="result")

    print_message(f"Prepared {total} chunks in {settings.chunks_dir}", style="result")
