import argparse
import asyncio

from dotenv import load_dotenv

from hanuman.index import index
from hanuman.search import search


load_dotenv()


async def main() -> None:
    parser = argparse.ArgumentParser(description="Hanuman - 2-Step RAG Application")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("search", help="Search using RAG")

    index_parser = subparsers.add_parser("index", help="Index a text file")
    index_parser.add_argument("input", help="Input text file path")

    args = parser.parse_args()

    if args.command == "index":
        await index(input_file=args.input)
    elif args.command == "search":
        await search()


if __name__ == "__main__":
    asyncio.run(main())
