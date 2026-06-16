"""PDF parsing utilities using PyMuPDF."""
import fitz  # PyMuPDF
from pathlib import Path


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text content from a PDF file.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Extracted text as a single string.

    Raises:
        FileNotFoundError: If PDF file doesn't exist.
        Exception: If PDF cannot be parsed.
    """
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    try:
        doc = fitz.open(pdf_path)
        text_parts = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            text_parts.append(text)

        doc.close()

        full_text = "\n\n".join(text_parts)
        return full_text

    except Exception as e:
        raise Exception(f"Error parsing PDF {pdf_path}: {str(e)}")


def extract_text_from_multiple_pdfs(pdf_paths: list[str]) -> list[dict]:
    """Extract text from multiple PDF files.

    Args:
        pdf_paths: List of paths to PDF files.

    Returns:
        List of dicts with 'text' and 'source' keys.
    """
    results = []

    for pdf_path in pdf_paths:
        try:
            text = extract_text_from_pdf(pdf_path)
            results.append({
                'text': text,
                'source': str(pdf_path)
            })
        except Exception as e:
            print(f"Warning: Failed to extract text from {pdf_path}: {e}")

    return results
