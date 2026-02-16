import sys

from langchain_groq import ChatGroq

from hanuman.services.prompt import get_prompt
from hanuman.services.store import get_vector_store
from hanuman.settings import settings


async def search(query: str) -> None:
    if not settings.groq_api_key:
        print("Error: GROQ_API_KEY is not set")
        sys.exit(1)

    print(f"Searching for: {query}")

    vector_store = await get_vector_store()
    docs = vector_store.similarity_search(query, k=settings.top_k)

    if not docs:
        print("No results found")
        return

    context = "\n\n".join(doc.page_content for doc in docs)
    prompt = await get_prompt(name="search", query=query, context=context)

    llm = ChatGroq(
        model=settings.chat.model,
        groq_api_key=settings.groq_api_key.get_secret_value() if settings.groq_api_key else None,  # type: ignore[arg-type]
        temperature=settings.chat.temperature,
        max_tokens=settings.chat.max_tokens,
    )

    response = llm.invoke(prompt)
    print("\n--- Response ---")
    print(response.content)
