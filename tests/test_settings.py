from unittest.mock import patch


class TestSettings:
    def test_default_database_settings(self):
        from hanuman.settings import DatabaseSettings

        db = DatabaseSettings()
        assert db.host == "localhost"
        assert db.port == 5432
        assert db.user == "hanuman"
        assert db.database == "hanuman"

    def test_database_connection_string(self):
        from hanuman.settings import DatabaseSettings

        db = DatabaseSettings()
        expected = "postgresql+psycopg://hanuman:hanuman@localhost:5432/hanuman"
        assert db.connection_string == expected

    def test_custom_database_settings(self):
        from hanuman.settings import DatabaseSettings

        db = DatabaseSettings(host="db.example.com", port=5433, user="admin", password="secret", database="mydb")  # nosec: B106
        expected = "postgresql+psycopg://admin:secret@db.example.com:5433/mydb"
        assert db.connection_string == expected

    def test_embedding_settings(self):
        from hanuman.settings import EmbeddingSettings

        emb = EmbeddingSettings()
        assert emb.model == "intfloat/multilingual-e5-large"
        assert emb.query_prefix == "query: "
        assert emb.document_prefix == "passage: "

    def test_chat_settings(self):
        from hanuman.settings import ChatSettings

        chat = ChatSettings()
        assert chat.model == "meta-llama/llama-3.1-8b-instruct"
        assert chat.temperature == 0.0
        assert chat.max_tokens == 2048

    @patch.dict("os.environ", {"HANUMAN_OPENROUTER_API_KEY": "test_key"}, clear=False)
    def test_settings_from_env(self):
        from pydantic_settings import BaseSettings

        class TestSettings(BaseSettings):
            openrouter_api_key: str | None = None

            class Config:
                env_prefix = "HANUMAN_"

        settings = TestSettings()
        assert settings.openrouter_api_key == "test_key"

    def test_default_settings(self):
        from hanuman.settings import Settings

        settings = Settings()
        assert settings.chunk_size == 1000
        assert settings.chunk_overlap == 200
        assert settings.top_k == 4

    @patch.dict("os.environ", {"HF_TOKEN": "hf_test"}, clear=True)
    def test_hf_token_from_env(self):
        from hanuman.settings import Settings

        settings = Settings()
        assert settings.hf_token is not None
        assert settings.hf_token.get_secret_value() == "hf_test"

    @patch.dict("os.environ", {"HUGGINGFACEHUB_API_TOKEN": "hf_legacy"}, clear=True)
    def test_hf_token_legacy_env_name(self):
        from hanuman.settings import Settings

        settings = Settings()
        assert settings.hf_token is not None
        assert settings.hf_token.get_secret_value() == "hf_legacy"

    @patch.dict("os.environ", {"SOME_UNRELATED_KEY": "whatever"}, clear=False)
    def test_unknown_env_keys_are_ignored(self):
        from hanuman.settings import Settings

        assert Settings().top_k == 4

    async def test_embeddings_receive_hf_token(self):
        from unittest.mock import MagicMock, patch

        from hanuman.services.store import VectorStoreManager

        with patch("hanuman.services.store.HuggingFaceEmbeddings") as mock_emb:
            manager = VectorStoreManager()
            with (
                patch("hanuman.services.store.settings") as mock_settings,
                patch("hanuman.services.store.PGVector", MagicMock()),
            ):
                mock_settings.hf_token.get_secret_value.return_value = "hf_test"
                await manager.initialize()

        assert mock_emb.call_args.kwargs["model_kwargs"] == {"token": "hf_test"}

    async def test_embeddings_omit_token_when_unset(self):
        from unittest.mock import MagicMock, patch

        from hanuman.services.store import VectorStoreManager

        with patch("hanuman.services.store.HuggingFaceEmbeddings") as mock_emb:
            manager = VectorStoreManager()
            with (
                patch("hanuman.services.store.settings") as mock_settings,
                patch("hanuman.services.store.PGVector", MagicMock()),
            ):
                mock_settings.hf_token = None
                await manager.initialize()

        assert mock_emb.call_args.kwargs["model_kwargs"] == {}
