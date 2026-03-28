from pathlib import Path


NOTES_DIR = Path("Notes")


#this definition is used to list all the note files in the Notes directory
def list_note_files() -> list[str]: 
    return [file.name for file in NOTES_DIR.glob("*.txt")]

#this definition is used to read the content of a note file
def read_note_file(file_name: str) -> str:
    file_path = NOTES_DIR / file_name 
    return file_path.read_text(encoding="utf-8") 

 #this definition is used to save the output to a file
def save_output(content: str, filename: str = "output.txt") -> str:
    output_path = Path(filename)
    output_path.write_text(content, encoding="utf-8")
    return str(output_path)