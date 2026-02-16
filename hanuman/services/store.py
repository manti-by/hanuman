from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector
from sqlalchemy.engine import Engine

from hanuman.settings import settings


class VectorStoreManager:
    def __init__(self) -> None:
        self.embeddings: HuggingFaceEmbeddings | None = None
        self.vector_store: PGVector | None = None

    async def initialize(self) -> PGVector:
        self.embeddings = HuggingFaceEmbeddings(model=settings.embedding.model)
        self.vector_store = PGVector(
            embeddings=self.embeddings,
            connection=settings.database.connection_string,
            collection_name="documents",
        )
        return self.vector_store

    def close(self):
        del self.embeddings

        if self.vector_store and isinstance(self.vector_store._engine, Engine):
            self.vector_store._engine.dispose()
        self.vector_store = None


@asynccontextmanager
async def get_vector_store_context() -> AsyncGenerator[PGVector]:
    manager = VectorStoreManager()
    try:
        store = await manager.initialize()
        yield store
    finally:
        manager.close()
