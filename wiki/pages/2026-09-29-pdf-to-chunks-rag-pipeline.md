---
title: PDF-to-chunks RAG pipeline
date: 2026-09-29
type: implementation
status: resolved
session_id: ses_f1666d8efffeePzGnkKPZn63Uh
services: [prepare, index, search]
branch: master
tickets: []
tags: [rag, pdf, pymupdf4llm, markdown, chunking, embeddings, pgvector]
related: [2026-09-25-groq-to-openrouter-migration]
---

# PDF-to-chunks RAG pipeline

## TL;DR

Added a `prepare` command that converts every PDF in `data/raw/` to markdown with
`pymupdf4llm` and splits it into header-aware chunks in `data/chunks/`, then rewrote
`index` to parse those chunks (front matter + markdown) instead of plain text. The corpus
is a 623-page Belarusian technical code (`data/raw/TKP339-2022.pdf`); 1654 chunks are
indexed and searchable. Embeddings moved from `all-MiniLM-L6-v2` to
`intfloat/multilingual-e5-large` with the `query:`/`passage:` prefixes E5 requires —
MiniLM's 256-token limit was silently truncating roughly half of every Russian chunk.

---

## Overview

The pipeline before this session was `index <text-file>` → `TextLoader` →
`RecursiveCharacterTextSplitter` → pgvector. It could not ingest the project's actual
source material, a set of large scanned-formatted standards PDFs.

The new flow is three commands:

- `make prepare` — PDF → `data/markdown/*.md` → `data/chunks/<doc>/*.md`
- `make index` — `data/chunks/**/*.md` → pgvector
- `make truncate` — empty `langchain_pg_embedding`

Key outcomes:

- **1654 chunks** from 623 pages, median 1808 bytes, max 4682 bytes, 0 empty.
- **414 headings** recovered, with correct hierarchy restored (see Step 1).
- **Retrieval verified** on Russian queries — all three probes returned on-topic sections.
- `make check` (ty, ruff, bandit, pre-commit) and **51 tests** pass.

---

## Step 1 — Heading detection required PDF forensics, not a regex guess

The task was "split on headers, maybe use regular expressions, they look like `4.4.3.3` or
`3.131`". Inspecting the PDF first changed the design in three ways.

**File:** `hanuman/prepare.py:14-16`

```python
HEADING_RE = re.compile(r"^(?P<hashes>#{1,6})\s+(?P<text>.+?)\s*$")
SECTION_RE = re.compile(r"^(?P<number>\d+(?:\.\d+)*)\.?(?=\s|$)")
CAPTION_RE = re.compile(r"^(?:Таблица|Рисунок|Рис\.)\b")
```

### Ruled out: a number regex alone

Font metadata showed headings are bold, but body text also begins with numbers —
`50 В переменного и 120 В постоянного тока.` matches `^\d+` yet is not a heading. Bold is
the real signal, and `pymupdf4llm` already emits markdown headers on that basis, so its
output was kept as the base rather than re-derived from raw text.

### Ruled out: `3.131` as a heading

Section 3 is the glossary, and its terms are bold lines where **the definition sits on the
same line** as the term:

```markdown
**3.1 арматура линейная на воздушной линии электропередачи напряжением до 35 кВ:** Устройство, предназначенное для подвешивания и крепления...
```

Promoting those to headings would move each definition into the header and leave an empty
chunk. 131 terms exist. They are deliberately **left as body text** — they never reach
`normalize_headings` as headings because `pymupdf4llm` does not mark them.

### Ruled out: trusting pymupdf4llm's heading levels

`pymupdf4llm` flattens everything below level 2 to `###`, so `4.2.1` and `4.4.3.3` become
indistinguishable. Level is therefore re-derived from the number's dot depth:

**File:** `hanuman/prepare.py:32-36`

```python
def _heading_level(heading: str, fallback: int) -> int:
    match = SECTION_RE.match(heading)
    if match:
        return min(match["number"].count(".") + 1, MAX_HEADING_LEVEL)
    return min(fallback, MAX_HEADING_LEVEL)
```

### What else the PDF analysis caught

Three classes of false-positive heading, all fixed in `normalize_headings`
(`hanuman/prepare.py:64-91`):

- **Wrapped heading continuations** (~80 cases). A long heading wrapping across a line
  produced a second `###` for the tail — `### **частотой 50 Гц:**` following
  `4.4.9.4 Испытание изоляции повышенным напряжением`. A non-numbered heading directly
  after a heading is now merged into it.
- **Hyphenation artifacts.** `характери-` + `стик` must become `характеристик`, not
  `характери- стик` (`hanuman/prepare.py:39-42`).
- **Table captions** emitted as `####`. `Таблица 4.4.14` / `Рисунок 6.2.13` are demoted to
  bold text so they no longer fragment chunks.

Running header `**ТКП 339-2022**` (617 occurrences) and standalone page numbers are stripped
by `strip_page_furniture` (`hanuman/prepare.py:45-61`). One page-boundary artifact
(`Издание официальное` bleeding into section 1) survived: it appears twice in 623 pages, so
no safe generic rule removes it without risking real content.

### Added beyond the request: a size bound

`MarkdownHeaderTextSplitter` alone produced a **117 KB chunk** against a 1808-byte median,
for sections with no sub-headings. A `RecursiveCharacterTextSplitter` pass using the existing
`settings.chunk_size` / `chunk_overlap` bounds each piece. This raised the count from 395 to
1654 and capped size at ~1000 characters. Flagged to the user as reversible.

---

## Step 2 — `index` reads markdown with front matter

`TextLoader` is gone. `parse_markdown` splits front matter off with a regex and parses it
with `yaml.safe_load` (already a transitive dependency — the `frontmatter` package is not
installed, and `langchain_community.FrontMatterLoader` no longer exists upstream).

**File:** `hanuman/index.py:29-39`

```python
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
```

`index` takes an optional path defaulting to `settings.chunks_dir`, so a file and a
directory share one code path. Dropping `TextLoader` removed the last
`langchain_community` import in the codebase.

### Embedding model: truncation was the real defect

`all-MiniLM-L6-v2` caps at 256 tokens. Median chunks are 839 characters and Cyrillic
tokenises at roughly 2 characters per token, so about half of each chunk was being cut off
before embedding — invisibly.

**File:** `hanuman/settings.py:19-22`

```python
class EmbeddingSettings(BaseModel):
    model: str = "intfloat/multilingual-e5-large"
    document_prefix: str = "passage: "
    query_prefix: str = "query: "
```

E5 is trained with asymmetric prefixes, so `index` prepends `document_prefix` to every
chunk and `search` prepends `query_prefix` to the query
(`hanuman/search.py:50`). **This was added beyond the literal request** because swapping in
an E5 model without prefixes degrades retrieval in a way that reads as "the new model is
worse" rather than as a setup bug. Both prefixes are env-overridable.

### `make truncate`

```makefile
truncate:
	uv run python -c "...TRUNCATE langchain_pg_embedding..."
```

Reports the deleted row count rather than running silently. Kept as a self-contained
one-liner so destructive SQL stays visible at the point of invocation, instead of a
`main.py truncate` subcommand.

---

## Test Results

- `make check` — ty, ruff, ruff-format, pyupgrade, bandit, pre-commit hooks: all pass.
- `uv run pytest tests/` — **51 passed** (37 pre-existing + 14 new).
- New coverage: `tests/test_prepare.py` (18 tests) for page furniture, heading depth,
  wrapped/­hyphenated merges, caption demotion, chunk bounding; `tests/test_main.py` for
  markdown stripping, front-matter parsing, empty-chunk skipping, both E5 prefixes, and
  the missing-path / empty-directory exits.

Two real bugs were caught by tests written after the first implementation, not by review:

- `HEADERS_TO_SPLIT_ON` was built as `f"#{level}"`, producing `'#1'`, `'#2'`. The splitter
  silently returned **1 document with empty metadata** for the whole 1.2 MB file instead of
  failing — the first end-to-end run reported "1 chunk(s)".
- Stripping a table's leading `|` left the following space, yielding `' 750 В 10 А'`.

End-to-end verification against the real database (1654 rows indexed, then similarity
search):

- **испытание изоляции обмотки статора** → `4.4.3.4 Испытание изоляции повышенным напряжением частотой 50 Гц` (both hits)
- **требования к заземляющим проводникам** → `4.4.19.7 Проверка сопротивления заземляющих устройств`, `4.3.12 Заземляющие проводники`
- **учет активной электроэнергии на промышленных предприятиях** → `4.2.2 Пункты установки средств учета электроэнергии`, `4.2.3`

---

## Follow-ups

- **`langchain-community` is now an unused dependency.** It is still declared in
  `pyproject.toml` but nothing imports it. Left in place deliberately — removing a
  dependency is the maintainer's call.
- **Chunks sit near the e5-large 512-token ceiling.** `CHUNK_SIZE=1000` is *characters*, and
  Cyrillic runs ~2 chars/token, so the largest chunks land near 500 tokens. It fits, with
  little headroom. If retrieval misses on the biggest chunks, drop `CHUNK_SIZE` to ~700 and
  re-run `prepare` then `index`.
- **Re-indexing costs ~8m40s** on CPU with e5-large (1m23s once the model is warm). Painful
  to re-run often.
- **`make truncate` wipes every collection**, not just `documents` — `langchain_pg_embedding`
  is PGVector's single shared table. Harmless while there is one collection.
- **`index` appends, it does not replace.** Run `make truncate` before re-indexing or rows
  accumulate.
- **Glossary terms (3.x) are not individually retrievable** — they stay inside the section 3
  chunk. A future term-level index would need the term text split from its definition.
- **`uv add pymupdf4llm` bumped transitive deps**: `openai` 2.54.0 → 3.19.2 and
  `filelock` 3.32.7 → 4.0.3. Worth reviewing before merge since `langchain-openai` depends
  on `openai`.
- **Work is uncommitted on `master`.** Per `AGENTS.md` this should move to a feature branch
  (`opencode/feature/...`) and be opened as a PR rather than committed to `master`.
- **One page-boundary artifact remains** (`Издание официальное` bleeding into section 1) —
  too rare to fix without a rule that could delete real content.

## References

- Related: [[2026-09-25-groq-to-openrouter-migration]]
- Corpus: `data/raw/TKP339-2022.pdf` — 623 pages, Belarusian technical code
- External: https://huggingface.co/intfloat/multilingual-e5-large
