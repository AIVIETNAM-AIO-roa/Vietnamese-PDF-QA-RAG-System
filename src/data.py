from enum import Enum
class DocumentType(Enum):
    TXT = "txt"
    MARKDOWN = "markdown"
    PDF = "pdf"
    HTML = "html"

class Document():
    def __init__(self, doc_id, file_path, doc_type, content):
        self.doc_id = doc_id
        self.file_path = file_path
        self.doc_type = doc_type
        self.content = content

class Chunk():
    def __init__(self, chunk_id, doc_id, content, start, end, index):
        self.chunk_id = chunk_id
        self.doc_id = doc_id
        self.content = content
        self.start_index = start
        self.end_index = end
        self.metadata = {
            "chunk_index": index,
            "chunk_length": len(content),
            "strategy": "fixed_size"
        }