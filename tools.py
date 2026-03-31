from pathlib import Path
import pypdf
from docx import Document
from pptx import Presentation


NOTES_DIR = Path("Notes")
SUPPORTED_EXTENSIONS = ["*.txt", "*.pdf", "*.docx", "*.pptx"]


def set_notes_dir(path: str):
    global NOTES_DIR
    NOTES_DIR = Path(path)


def list_note_files() -> list[str]:
    files = []
    for ext in SUPPORTED_EXTENSIONS:
        files.extend([f.name for f in NOTES_DIR.glob(ext)])
    return files


def read_note_file(file_name: str) -> str:
    file_path = NOTES_DIR / file_name
    suffix = file_path.suffix.lower()

    if suffix == ".txt":
        return file_path.read_text(encoding="utf-8")

    elif suffix == ".pdf":
        reader = pypdf.PdfReader(str(file_path))
        return "\n".join(
            page.extract_text() for page in reader.pages if page.extract_text()
        )

    elif suffix == ".docx":
        doc = Document(str(file_path))
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())

    elif suffix == ".pptx":
        prs = Presentation(str(file_path))
        text = []
        for slide in prs.slides:
            for shape in slide.shapes:
                if shape.has_text_frame:
                    for para in shape.text_frame.paragraphs:
                        if para.text.strip():
                            text.append(para.text)
        return "\n".join(text)

    else:
        return f"Unsupported file format: {suffix}"


def save_output(content: str, filename: str = "output.txt") -> str:
    output_path = Path(filename)
    output_path.write_text(content, encoding="utf-8")
    return str(output_path)