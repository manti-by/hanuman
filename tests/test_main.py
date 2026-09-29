from contextlib import asynccontextmanager
from unittest.mock import MagicMock, patch

import pytest

from hanuman.index import index, parse_markdown, strip_markdown
from hanuman.search import search
from hanuman.services.utils import clean_text


class TestStripMarkdown:
    def test_removes_heading_markers(self):
        assert strip_markdown("## 4.1 Общие положения") == "4.1 Общие положения"

    def test_removes_emphasis_and_code_marks(self):
        assert strip_markdown("**4.1.1** Требования `кода`") == "4.1.1 Требования кода"

    def test_flattens_table_rows(self):
        markdown = "| Напряжение | Ток |\n|---|---|\n| 750 В | 10 А |"
        assert strip_markdown(markdown) == "Напряжение Ток\n750 В 10 А"

    def test_drops_table_delimiter_row(self):
        assert "---" not in strip_markdown("| a | b |\n|---|---|\n| 1 | 2 |")

    def test_keeps_plain_text(self):
        assert strip_markdown("Обычный текст раздела.") == "Обычный текст раздела."


class TestParseMarkdown:
    def test_reads_front_matter_into_metadata(self, tmp_path):
        path = tmp_path / "0001-4.md"
        path.write_text("---\nsource: manual\nheaders:\n- 4 Общие правила\n---\n\nТекст.", encoding="utf-8")
        document = parse_markdown(path)
        assert document.metadata["source"] == "manual"
        assert document.metadata["headers"] == ["4 Общие правила"]
        assert document.page_content == "Текст."

    def test_handles_file_without_front_matter(self, tmp_path):
        path = tmp_path / "plain.md"
        path.write_text("Без метаданных.", encoding="utf-8")
        document = parse_markdown(path)
        assert document.page_content == "Без метаданных."
        assert document.metadata == {"file": "plain.md"}

    def test_handles_empty_front_matter(self, tmp_path):
        path = tmp_path / "empty.md"
        path.write_text("---\nsource: manual\nheaders:\n\n---\n\nТекст.", encoding="utf-8")
        assert parse_markdown(path).page_content == "Текст."


class TestCleanText:
    def test_remove_multiple_newlines(self):
        text = "line1\n\n\nline2\n\nline3"
        result = clean_text(text)
        assert "\n\n\n" not in result

    def test_remove_multiple_spaces(self):
        text = "word1    word2"
        result = clean_text(text)
        assert "    " not in result

    def test_remove_trailing_whitespace(self):
        text = "line1   \nline2\t\t"
        result = clean_text(text)
        assert "   \n" not in result

    def test_strip_start_end(self):
        text = "   \ncontent\n   "
        result = clean_text(text)
        assert result.startswith("content")

    def test_empty_string(self):
        result = clean_text("")
        assert result == ""

    def test_preserve_normal_structure(self):
        text = "line1\nline2"
        result = clean_text(text)
        assert result == "line1\nline2"


class TestMain:
    @pytest.mark.asyncio
    @patch("hanuman.index.get_vector_store_context")
    async def test_index_mode_defaults_to_chunks_dir(self, mock_get_store_ctx, tmp_path):
        chunks_dir = tmp_path / "chunks" / "manual"
        chunks_dir.mkdir(parents=True)
        (chunks_dir / "0001-4.md").write_text(
            "---\nsource: manual\nheaders:\n- 4 Общие правила\n---\n\n**4.1** Текст раздела.",
            encoding="utf-8",
        )

        mock_store = MagicMock()

        @asynccontextmanager
        async def mock_context():
            yield mock_store

        mock_get_store_ctx.return_value = mock_context()

        with patch("hanuman.index.settings") as mock_settings:
            mock_settings.chunks_dir = tmp_path / "chunks"
            mock_settings.embedding.document_prefix = "passage: "
            await index()

        mock_store.add_documents.assert_called_once()
        documents = mock_store.add_documents.call_args[0][0]
        assert documents[0].page_content == "passage: 4.1 Текст раздела."
        assert documents[0].metadata["source"] == "manual"
        assert documents[0].metadata["file"] == "0001-4.md"

    @pytest.mark.asyncio
    @patch("hanuman.index.get_vector_store_context")
    async def test_index_mode_explicit_path(self, mock_get_store_ctx, tmp_path):
        (tmp_path / "note.md").write_text("Одиночный файл.", encoding="utf-8")

        mock_store = MagicMock()

        @asynccontextmanager
        async def mock_context():
            yield mock_store

        mock_get_store_ctx.return_value = mock_context()

        await index(input_path=str(tmp_path))

        documents = mock_store.add_documents.call_args[0][0]
        assert documents[0].page_content == "passage: Одиночный файл."

    @pytest.mark.asyncio
    @patch("hanuman.index.get_vector_store_context")
    async def test_index_mode_missing_path(self, mock_get_store_ctx, tmp_path):
        with pytest.raises(SystemExit) as exc_info:
            await index(input_path=str(tmp_path / "nope"))
        assert exc_info.value.code == 1

    @pytest.mark.asyncio
    @patch("hanuman.index.get_vector_store_context")
    async def test_index_mode_empty_directory(self, mock_get_store_ctx, tmp_path):
        (tmp_path / "chunks").mkdir()
        with pytest.raises(SystemExit) as exc_info:
            await index(input_path=str(tmp_path / "chunks"))
        assert exc_info.value.code == 1

    @pytest.mark.asyncio
    @patch("hanuman.index.get_vector_store_context")
    async def test_index_mode_skips_empty_chunks(self, mock_get_store_ctx, tmp_path):
        (tmp_path / "a.md").write_text("---\nsource: s\n---\n\n", encoding="utf-8")
        (tmp_path / "b.md").write_text("Содержимое.", encoding="utf-8")

        mock_store = MagicMock()

        @asynccontextmanager
        async def mock_context():
            yield mock_store

        mock_get_store_ctx.return_value = mock_context()

        await index(input_path=str(tmp_path))

        documents = mock_store.add_documents.call_args[0][0]
        assert [document.metadata["file"] for document in documents] == ["b.md"]

    @pytest.mark.asyncio
    @patch("hanuman.index.get_vector_store_context")
    async def test_index_mode_applies_document_prefix(self, mock_get_store_ctx, tmp_path):
        (tmp_path / "a.md").write_text("Содержимое.", encoding="utf-8")

        mock_store = MagicMock()

        @asynccontextmanager
        async def mock_context():
            yield mock_store

        mock_get_store_ctx.return_value = mock_context()

        await index(input_path=str(tmp_path))

        documents = mock_store.add_documents.call_args[0][0]
        assert documents[0].page_content == "passage: Содержимое."

    @pytest.mark.asyncio
    @patch("hanuman.search.ChatOpenAI")
    @patch("hanuman.search.get_vector_store_context")
    async def test_search_applies_query_prefix(self, mock_get_store_ctx, mock_chat):
        mock_store = MagicMock()
        mock_store.similarity_search.return_value = [MagicMock(page_content="passage: контекст")]

        @asynccontextmanager
        async def mock_context():
            yield mock_store

        mock_get_store_ctx.return_value = mock_context()

        with patch("hanuman.search.settings") as mock_settings:
            mock_settings.openrouter_api_key = MagicMock()
            mock_settings.top_k = 4
            mock_settings.chat.model = "test"
            mock_settings.chat.temperature = 0.0
            mock_settings.chat.max_tokens = 16
            mock_settings.embedding.query_prefix = "query: "
            with patch("builtins.input", side_effect=["проводники", "2"]):
                await search()

        mock_store.similarity_search.assert_called_once_with("query: проводники", k=4)

    @pytest.mark.asyncio
    @patch("hanuman.search.get_vector_store_context")
    @patch("hanuman.search.ChatOpenAI")
    async def test_search_mode_no_api_key(self, mock_chat, mock_get_store_ctx):
        with patch("hanuman.search.settings") as mock_settings:
            mock_settings.openrouter_api_key = None

            with pytest.raises(SystemExit) as exc_info:
                await search()
            assert exc_info.value.code == 1

    @pytest.mark.asyncio
    @patch("hanuman.search.get_vector_store_context")
    @patch("hanuman.search.ChatOpenAI")
    async def test_search_mode(self, mock_chat_class, mock_get_store_ctx):
        with patch("hanuman.search.settings") as mock_settings:
            mock_api_key = MagicMock()
            mock_api_key.get_secret_value.return_value = "test_key"
            mock_settings.openrouter_api_key = mock_api_key

            mock_store = MagicMock()
            mock_store.similarity_search.return_value = [MagicMock(page_content="relevant context")]

            @asynccontextmanager
            async def mock_context():
                yield mock_store

            mock_get_store_ctx.return_value = mock_context()

            mock_response = MagicMock()
            mock_response.content = "Test response"
            mock_chat = MagicMock()
            mock_chat.invoke.return_value = mock_response
            mock_chat_class.return_value = mock_chat

            with patch("builtins.input", side_effect=["test query", "2"]):
                await search()

            mock_store.similarity_search.assert_called()
            mock_chat_class.assert_called()
