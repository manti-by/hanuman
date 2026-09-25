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
        assert emb.model == "sentence-transformers/all-MiniLM-L6-v2"

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
