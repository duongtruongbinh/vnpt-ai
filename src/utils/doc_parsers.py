"""Document parsing utilities for various file formats."""

from pathlib import Path

from src.utils.common import normalize_text
from src.utils.logging import print_log


def load_pdf(file_path: Path) -> str:
    """Load text from PDF file."""
    try:
        import pypdf
    except ImportError:
        raise ImportError("pypdf is required for PDF files. Install with: pip install pypdf")

    reader = pypdf.PdfReader(str(file_path))
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text.strip()


def load_docx(file_path: Path) -> str:
    """Load text from DOCX file."""
    try:
        import docx
    except ImportError:
        raise ImportError("python-docx is required for DOCX files. Install with: pip install python-docx")

    doc = docx.Document(str(file_path))
    return "\n".join([para.text for para in doc.paragraphs])


def load_txt(file_path: Path) -> str:
    """Load text from TXT file."""
    with open(file_path, encoding="utf-8") as f:
        return f.read()


def load_document(file_path: Path) -> tuple[str | None, dict | None]:
    """Load document (PDF, DOCX, TXT), normalize text, and return (text, metadata).

    Returns (None, None) for unsupported or failed files.
    """
    ext = file_path.suffix.lower()

    try:
        if ext == ".pdf":
            text = load_pdf(file_path)
        elif ext == ".docx":
            text = load_docx(file_path)
        elif ext == ".txt":
            text = load_txt(file_path)
        else:
            return None, None

        text = normalize_text(text)
        if not text:
            return None, None

        metadata = {
            "source_file": str(file_path),
            "file_name": file_path.name,
            "file_type": ext[1:],
        }
        return text, metadata

    except Exception as e:
        print_log(f"        [Error] Failed to load {file_path.name}: {e}")
        return None, None
