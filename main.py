import argparse
import asyncio
from argparse import Namespace

from hanuman.index import index
from hanuman.search import search


parser = argparse.ArgumentParser(description="Hanuman - 2-Step RAG Application")
subparsers = parser.add_subparsers(dest="command", required=True)
subparsers.add_parser("search", help="Search using RAG")

index_parser = subparsers.add_parser("index", help="Index a text file")
index_parser.add_argument("input", help="Input text file path")


async def main(args: Namespace) -> None:
    if args.command == "index":
        await index(input_file=args.input)
    elif args.command == "search":
        await search()


if __name__ == "__main__":
    args = parser.parse_args()
    asyncio.run(main(args=args))
