from pathlib import Path

from pydantic import BaseModel, SecretStr
from pydantic_settings import BaseSettings


class DatabaseSettings(BaseModel):
    host: str = "localhost"
    port: int = 5432
    user: str = "hanuman"
    password: str = "hanuman"
    database: str = "hanuman"

    @property
    def connection_string(self) -> str:
        return f"postgresql+psycopg://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


class EmbeddingSettings(BaseModel):
    model: str = "sentence-transformers/all-MiniLM-L6-v2"


class ChatSettings(BaseModel):
    model: str = "meta-llama/llama-3.1-8b-instruct"
    temperature: float = 0.0
    max_tokens: int = 2048


class Settings(BaseSettings):
    database: DatabaseSettings = DatabaseSettings()
    embedding: EmbeddingSettings = EmbeddingSettings()
    chat: ChatSettings = ChatSettings()

    base_path: Path = Path(__file__).resolve().parent.parent
    openrouter_api_key: SecretStr | None = None
    huggingfacehub_api_token: SecretStr | None = None
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 4

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        env_nested_delimiter = "__"


settings = Settings()
