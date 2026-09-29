import argparse
import asyncio
from argparse import Namespace

from hanuman.index import index
from hanuman.prepare import prepare
from hanuman.search import search


parser = argparse.ArgumentParser(description="Hanuman - 2-Step RAG Application")
subparsers = parser.add_subparsers(dest="command", required=True)
subparsers.add_parser("search", help="Search using RAG")
subparsers.add_parser("prepare", help="Convert raw PDFs to markdown and split them into chunks")

index_parser = subparsers.add_parser("index", help="Index markdown chunks")
index_parser.add_argument("input", nargs="?", help="Markdown file or directory (default: data/chunks)")


async def main(args: Namespace) -> None:
    if args.command == "index":
        await index(input_path=args.input)
    elif args.command == "search":
        await search()
    elif args.command == "prepare":
        await prepare()


if __name__ == "__main__":
    args = parser.parse_args()
    asyncio.run(main(args=args))
