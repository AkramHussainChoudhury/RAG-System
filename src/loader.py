from pathlib import Path
from pypdf import PdfReader
from src.tracer import get_logger

logger = get_logger("loader")


def load_pdf(path: str) -> str:
    """Extract all text from a PDF file, page by page."""
    file = Path(path)
    if not file.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    reader = PdfReader(str(file))
    pages_text = []

    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages_text.append(text)
        logger.info(f"  Page {i + 1}/{len(reader.pages)}: {len(text)} chars")

    full_text = "\n".join(pages_text)
    logger.info(f"Total extracted: {len(full_text)} chars from {len(reader.pages)} pages")
    return full_text
