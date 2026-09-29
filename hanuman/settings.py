from pathlib import Path

from pydantic import AliasChoices, BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


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
    model: str = "intfloat/multilingual-e5-large"
    document_prefix: str = "passage: "
    query_prefix: str = "query: "


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
    hf_token: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("HF_TOKEN", "HUGGINGFACEHUB_API_TOKEN"),
    )
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 4

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )

    @property
    def raw_dir(self) -> Path:
        return self.base_path / "data" / "raw"

    @property
    def markdown_dir(self) -> Path:
        return self.base_path / "data" / "markdown"

    @property
    def chunks_dir(self) -> Path:
        return self.base_path / "data" / "chunks"


settings = Settings()
