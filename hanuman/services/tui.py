import aiofiles
from rich.console import Console
from rich.text import Text


console = Console()


def print_message(message: str, style: str | None = None):
    if style == "heading":
        console.print("\n\u25cf ", style="bold bright_green", end="")
        console.print(message, style="bold bright_white")
    elif style == "result":
        console.print("→ ", style="bold bright_green", end="")
        console.print(message, style="white")
    elif style == "info":
        console.print(message, style="bright_black")
    elif style == "error":
        console.print(message, style="red")
    else:
        console.print(message)


async def print_heading():
    async with aiofiles.open("tui/header.txt") as file:
        text = await file.read()

    text = Text(text)
    text.stylize("magenta", 0, 250)
    text.stylize("cyan", 250, 500)
    text.stylize("blue", 500, 750)

    console.print()
    console.print(text, end="")
