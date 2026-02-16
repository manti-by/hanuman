import aiofiles

from hanuman.settings import settings


async def get_prompt(name: str, **kwargs) -> str:
    async with aiofiles.open(settings.base_path / f"hanuman/services/prompts/{name}.md") as file:
        content = await file.read()
    if kwargs:
        return content.format(**kwargs)
    return content
