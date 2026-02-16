from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hanuman.index import index
from hanuman.search import search
from hanuman.services.utils import clean_text


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
    @patch("hanuman.index.get_vector_store", new_callable=AsyncMock)
    @patch("hanuman.index.TextLoader")
    @patch("hanuman.index.Path")
    async def test_index_mode(self, mock_path_cls, mock_loader_class, mock_get_store):
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path_cls.return_value = mock_path_instance
        mock_doc = MagicMock()
        mock_doc.page_content = "test content"
        mock_doc.metadata = {}
        mock_loader = MagicMock()
        mock_loader.load.return_value = [mock_doc]
        mock_loader_class.return_value = mock_loader

        mock_store = MagicMock()
        mock_get_store.return_value = mock_store

        await index("/fake/path.txt")

        mock_loader.load.assert_called_once()
        mock_store.add_documents.assert_called_once()

    @pytest.mark.asyncio
    @patch("hanuman.search.get_vector_store", new_callable=AsyncMock)
    @patch("hanuman.search.ChatGroq")
    async def test_search_mode_no_api_key(self, mock_chat, mock_get_store):
        with patch("hanuman.search.settings") as mock_settings:
            mock_settings.groq_api_key = None

            with pytest.raises(SystemExit) as exc_info:
                await search()
            assert exc_info.value.code == 1

    @pytest.mark.asyncio
    @patch("hanuman.search.get_vector_store", new_callable=AsyncMock)
    @patch("hanuman.search.ChatGroq")
    async def test_search_mode(self, mock_chat_class, mock_get_store):
        with patch("hanuman.search.settings") as mock_settings:
            mock_api_key = MagicMock()
            mock_api_key.get_secret_value.return_value = "test_key"
            mock_settings.groq_api_key = mock_api_key

            mock_store = MagicMock()
            mock_store.similarity_search.return_value = [MagicMock(page_content="relevant context")]
            mock_get_store.return_value = mock_store

            mock_response = MagicMock()
            mock_response.content = "Test response"
            mock_chat = MagicMock()
            mock_chat.invoke.return_value = mock_response
            mock_chat_class.return_value = mock_chat

            with patch("builtins.input", side_effect=["test query", "2"]):
                await search()

            mock_store.similarity_search.assert_called()
            mock_chat_class.assert_called()
