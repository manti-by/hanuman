import sys

from langchain_groq import ChatGroq

from hanuman.services.prompt import get_prompt
from hanuman.services.store import get_vector_store
from hanuman.services.tui import print_message
from hanuman.settings import settings


async def search(query: str) -> None:
    if not settings.groq_api_key:
        print_message("Error: GROQ_API_KEY is not set", style="error")
        sys.exit(1)

    print_message(f"Searching for: {query}", style="heading")
    vector_store = await get_vector_store()
    docs = vector_store.similarity_search(query, k=settings.top_k)
    if not docs:
        print_message("No results found", style="result")
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
    print_message("Search results:", style="result")
    if isinstance(response.content, str):
        print_message(response.content, style="result")
    elif isinstance(response.content, list):
        for item in response.content:
            print_message(str(item), style="result")
    elif isinstance(response.content, dict):
        for key, value in response.content.items():
            print_message(f"{key}: {value}", style="result")
