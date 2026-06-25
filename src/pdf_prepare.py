import os
import hashlib
from src.data import Document, DocumentType
from typing import List

class PDFLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self) -> str:
        try:
            from pypdf import PdfReader
        except ImportError as e:
            raise ImportError("pypdf chưa được cài — chạy: pip install pypdf") from e
        
        texts = []
        reader = PdfReader(self.file_path)
        for page in reader.pages:
            text = page.extract_text() # <-- Đã sửa lỗi tại đây
            texts.append(text)
        return "\n\n".join(texts)

class TEXTLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self) -> str:
        with open(self.file_path, "r", encoding="utf-8") as f:
            return f.read()

SUPPORTED_LOADERS = {
    '.pdf': PDFLoader,
    '.txt': TEXTLoader,
    '.md': TEXTLoader
}

class DocumentLoader():
    def __init__(self):
        pass

    def load(self, file_path: str) -> Document:
        if not self.supports(file_path):
            ext = self._get_extension(file_path)
            raise ValueError(f"Định dạng file {ext} không được hỗ trợ")
            
        ext = self._get_extension(file_path)
        loader_class = SUPPORTED_LOADERS.get(ext)
        loader_instance = loader_class(file_path)
        content = loader_instance.load()
        
        doc_typemap = {
            ".txt": DocumentType.TXT,
            ".md": DocumentType.MARKDOWN,
            ".pdf": DocumentType.PDF,
            ".html": DocumentType.HTML,
        }
        doc_type = doc_typemap.get(ext, DocumentType.TXT)
        
        return Document(
            doc_id=self._generate_doc_id(file_path),
            file_path=file_path,
            doc_type=doc_type,
            content=content,
        )

    def load_directory(self, dir_path: str) -> List[Document]:
        documents: List[Document] = []
        for root, _, files in os.walk(dir_path):
            for name in files:
                file_path = os.path.join(root, name)
                if self.supports(file_path):
                    documents.append(self.load(file_path))
        return documents

    def supports(self, file_path: str) -> bool:
        ext = self._get_extension(file_path)
        return ext in SUPPORTED_LOADERS

    def _get_extension(self, file_path: str) -> str:
        return os.path.splitext(file_path)[1].lower()

    def _generate_doc_id(self, file_path: str) -> str:
        abs_path = os.path.abspath(file_path)
        return hashlib.sha256(abs_path.encode()).hexdigest()