from enum import Enum

class DocumentType(Enum):
    """Enum định nghĩa các định dạng loại tài liệu được hệ thống RAG hỗ trợ."""
    TXT = "txt"
    MARKDOWN = "markdown"
    PDF = "pdf"
    HTML = "html"


class Document():
    """Class đại diện cho một tệp tài liệu thô hoàn chỉnh sau khi được nạp vào hệ thống."""
    def __init__(self, doc_id: str, file_path: str, doc_type: DocumentType, content: str):
        """Khởi tạo một thực thể Document chứa toàn bộ nội dung và siêu dữ liệu của tệp."""
        self.doc_id = doc_id
        self.file_path = file_path
        self.doc_type = doc_type
        self.content = content


class Chunk():
    """Class đại diện cho một phân đoạn văn bản nhỏ."""
    def __init__(self, chunk_id: str, doc_id: str, content: str, start: int, end: int, index: int):
        """Khởi tạo một phân đoạn văn bản nhỏ kèm theo thông tin vị trí và siêu dữ liệu truy vết nguồn."""
        self.chunk_id = chunk_id
        self.doc_id = doc_id
        self.content = content
        self.start_index = start
        self.end_index = end
        
        # Từ điển lưu trữ siêu dữ liệu (metadata) hỗ trợ đắc lực cho khâu lọc nâng cao khi truy vấn ChromaDB
        self.metadata = {
            "chunk_index": index,           
            "chunk_length": len(content),   
            "strategy": "fixed_size"        
        }