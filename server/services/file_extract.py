from __future__ import annotations

from io import BytesIO
from pathlib import Path
import re


def _clean_text(text: str) -> str:
    out = (text or "").replace("\r", " ").replace("\t", " ")
    out = re.sub(r"\s+", " ", out)
    return out.strip()


def _extract_txt(data: bytes) -> str:
    try:
        return _clean_text(data.decode("utf-8"))
    except Exception:
        return _clean_text(data.decode("latin-1", errors="ignore"))


def _extract_pdf(data: bytes) -> str:
    try:
        from pypdf import PdfReader  # type: ignore
    except Exception as exc:
        raise RuntimeError("PDF support requires pypdf") from exc

    reader = PdfReader(BytesIO(data))
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return _clean_text("\n".join(pages))


def _extract_epub(data: bytes) -> str:
    try:
        from ebooklib import epub  # type: ignore
        from ebooklib import ITEM_DOCUMENT  # type: ignore
    except Exception as exc:
        raise RuntimeError("EPUB support requires ebooklib") from exc

    try:
        from bs4 import BeautifulSoup  # type: ignore
    except Exception as exc:
        raise RuntimeError("EPUB support requires beautifulsoup4") from exc

    book = epub.read_epub(BytesIO(data))
    parts: list[str] = []
    for item in book.get_items_of_type(ITEM_DOCUMENT):
        html = item.get_body_content() or b""
        soup = BeautifulSoup(html, "html.parser")
        parts.append(soup.get_text(" "))
    return _clean_text("\n".join(parts))


def extract_text_from_file(filename: str, data: bytes, content_type: str | None = None) -> str:
    ext = Path(filename or "").suffix.lower()
    ctype = (content_type or "").lower()

    if ext in (".txt", ".md") or ctype.startswith("text/"):
        return _extract_txt(data)
    if ext == ".pdf" or "pdf" in ctype:
        return _extract_pdf(data)
    if ext == ".epub" or "epub" in ctype:
        return _extract_epub(data)

    raise RuntimeError("Unsupported file type. Supported: .txt, .pdf, .epub")
