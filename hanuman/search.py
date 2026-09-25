import sys

from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI

from hanuman.services.prompt import get_prompt
from hanuman.services.store import get_vector_store_context
from hanuman.services.tui import console, print_message
from hanuman.settings import settings


def print_llm_response(response: AIMessage):
    if isinstance(response.content, str):
        print_message(response.content, style="result")
    elif isinstance(response.content, list):
        for item in response.content:
            print_message(str(item), style="result")
    elif isinstance(response.content, dict):
        for key, value in response.content.items():
            print_message(f"{key}: {value}", style="result")


async def search() -> None:
    print_message("Setting up environment", style="heading")

    if not settings.openrouter_api_key:
        print_message("Error: OPENROUTER_API_KEY is not set", style="error")
        sys.exit(1)

    llm = ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=settings.openrouter_api_key,
        model=settings.chat.model,
        temperature=settings.chat.temperature,
        max_tokens=settings.chat.max_tokens,
    )

    while True:
        print_message("Enter your search query:", style="heading")
        print_message("Example: 'How does temperature affect egg formation?'", style="info")
        console.print("→ ", style="bold bright_green", end="")
        query = input().strip()

        if not query:
            print_message("Please enter a valid query", style="error")
            continue

        print_message(f"Searching for: {query}", style="heading")
        async with get_vector_store_context() as vector_store:
            docs = vector_store.similarity_search(query.lower(), k=settings.top_k)

            if not docs:
                print_message("No results found", style="result")
            else:
                context = "\n\n".join(doc.page_content for doc in docs)
                prompt = await get_prompt(name="search", query=query, context=context)
                print_llm_response(response=llm.invoke(prompt))

        print_message("\nHow would you like to proceed?")
        print_message("  [1] new search - default")
        print_message("  [2] quit")

        if input("Action: ").strip() == "2":
            print_message("\nExiting the search, buy.", style="result")
            return
