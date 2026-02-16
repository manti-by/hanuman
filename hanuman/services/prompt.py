import aiofiles


async def get_prompt(name: str, **kwargs) -> str:
    async with aiofiles.open(f"prompts/{name}.md") as file:
        content = await file.read()
    if kwargs:
        return content.format(**kwargs)
    return content
