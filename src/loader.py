import os
import hashlib
from src.models import Document, DocumentType
from typing import List

class PDFLoader:
    """Bộ nạp và trích xuất nội dung từ định dạng tệp tin PDF."""
    def __init__(self, file_path: str):
        """Khởi tạo bộ nạp PDF với đường dẫn tệp cụ thể."""
        self.file_path = file_path

    def load(self) -> str:
        """Đọc tệp PDF và trích xuất toàn bộ nội dung văn bản từ tất cả các trang."""
        try:
            from pypdf import PdfReader
        except ImportError as e:
            raise ImportError("pypdf chưa được cài — chạy: pip install pypdf") from e
        
        texts = []
        reader = PdfReader(self.file_path)
        for page in reader.pages:
            text = page.extract_text() 
            if text:  
                texts.append(text)
        return "\n\n".join(texts)


class TEXTLoader:
    """Bộ nạp và trích xuất nội dung từ các định dạng tệp văn bản thuần túy (TXT, Markdown)."""

    def __init__(self, file_path: str):
        """Khởi tạo bộ nạp văn bản với đường dẫn tệp cụ thể."""
        self.file_path = file_path

    def load(self) -> str:
        """Đọc tệp văn bản thô sử dụng bảng mã mã hóa UTF-8."""
        with open(self.file_path, "r", encoding="utf-8") as f:
            return f.read()

# Bảng ánh xạ định nghĩa các lớp Loader tương ứng với từng phần mở rộng của tệp tin
SUPPORTED_LOADERS = {
    '.pdf': PDFLoader,
    '.txt': TEXTLoader,
    '.md': TEXTLoader
}

class DocumentLoader():
    """Lớp điều hướng trung tâm quản lý khâu tải, phân loại dữ liệu và đóng gói thành thực thể Document."""

    def __init__(self):
        """Khởi tạo bộ điều phối DocumentLoader."""
        pass

    def load(self, file_path: str) -> Document:
        """Tải một tệp tin đơn lẻ, tự động nhận diện loader thích hợp và đóng gói thành đối tượng Document."""
        if not self.supports(file_path):
            ext = self._get_extension(file_path)
            raise ValueError(f"Định dạng file {ext} không được hỗ trợ")
            
        ext = self._get_extension(file_path)
        loader_class = SUPPORTED_LOADERS.get(ext)
        loader_instance = loader_class(file_path)
        content = loader_instance.load()
        
        # Ánh xạ từ chuỗi phần mở rộng sang kiểu Enum DocumentType hệ thống
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
        """Duyệt đệ quy toàn bộ thư mục để tìm và nạp tất cả các tệp tin hợp lệ được hệ thống hỗ trợ."""
        documents: List[Document] = []
        for root, _, files in os.walk(dir_path):
            for name in files:
                file_path = os.path.join(root, name)
                if self.supports(file_path):
                    documents.append(self.load(file_path))
        return documents

    def supports(self, file_path: str) -> bool:
        """Kiểm tra xem định dạng tệp tin cho trước có nằm trong danh sách hệ thống hỗ trợ hay không."""
        ext = self._get_extension(file_path)
        return ext in SUPPORTED_LOADERS

    def _get_extension(self, file_path: str) -> str:
        """Hàm nội bộ: Trích xuất phần mở rộng (đuôi file) và chuyển về dạng chữ thường."""
        return os.path.splitext(file_path)[1].lower()

    def _generate_doc_id(self, file_path: str) -> str:
        """Hàm nội bộ: Tạo mã định danh băm duy nhất (SHA-256 ID) dựa trên đường dẫn tuyệt đối của tệp tin."""
        abs_path = os.path.abspath(file_path)
        return hashlib.sha256(abs_path.encode()).hexdigest()