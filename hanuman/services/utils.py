import re


def clean_text(text: str) -> str:
    text = re.sub(r"\n+", "\n", text)
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE)
    return text.strip()
