from unittest.mock import MagicMock, patch

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
    @patch("main.get_vector_store")
    @patch("main.TextLoader")
    @patch("main.Path.exists")
    def test_index_mode(self, mock_exists, mock_loader_class, mock_get_store):
        mock_exists.return_value = True
        mock_doc = MagicMock()
        mock_doc.page_content = "test content"
        mock_doc.metadata = {}
        mock_loader = MagicMock()
        mock_loader.load.return_value = [mock_doc]
        mock_loader_class.return_value = mock_loader

        mock_store = MagicMock()
        mock_get_store.return_value = mock_store

        index("/fake/path.txt")

        mock_loader.load.assert_called_once()
        mock_store.add_documents.assert_called_once()

    @patch("main.settings")
    @patch("main.get_vector_store")
    @patch("main.ChatGroq")
    def test_search_mode_no_api_key(self, mock_chat, mock_get_store, mock_settings):
        mock_settings.groq_api_key = None

        with patch("sys.exit") as mock_exit:
            search("test query")
            mock_exit.assert_called_with(1)

    @patch("main.settings")
    @patch("main.get_vector_store")
    @patch("main.ChatGroq")
    def test_search_mode(self, mock_chat_class, mock_get_store, mock_settings):
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

        search("test query")

        search("test query")

        mock_store.similarity_search.assert_called_once()
        mock_chat_class.assert_called_once()
