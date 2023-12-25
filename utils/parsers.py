import pdfplumber
from docx import Document

class Parser:

    def __init__(self,
                 path_to_file:str):
        self.path = path_to_file

    def parse(self):
        raise NotImplementedError
    

class PDFParser(Parser):

    def __init__(self,
                 path_to_file: str):
        super().__init__(path_to_file)

    def parse(self):
        text = ""
        with pdfplumber.open(self.path) as pdf:

            for page in pdf.pages:
                try:
                    text += page.extract_text() + "\n"
                except:
                    pass

        text = text.strip()
        return text
    
class DocxParser(Parser):

    def __init__(self,
                 path_to_file: str):
        super().__init__(path_to_file)

    def parse(self):

        text = ""

        doc = Document(self.path)  
        paragraphs = doc.paragraphs
        text = "\n".join([p.text for p in paragraphs]).strip()

        return text