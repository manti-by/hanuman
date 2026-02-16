from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector

from hanuman.settings import settings


async def get_vector_store() -> PGVector:
    embeddings = HuggingFaceEmbeddings(model=settings.embedding.model)

    return PGVector(
        embeddings=embeddings,
        connection=settings.database.connection_string,
        collection_name="documents",
    )
