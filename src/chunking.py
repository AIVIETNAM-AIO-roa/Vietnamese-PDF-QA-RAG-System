from src.models import Document, Chunk
from typing import List

class TextChunker():
    """ Chia nhỏ document theo fixed_size """
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        if chunk_size <= 0:
            raise ValueError("chunk_size phải lớn hơn 0")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap không được âm")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap phải nhỏ hơn chunk_size")
            
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, document: Document) -> List[Chunk]:
        if not document.content:
            return []
        return self.chunk_by_fixed_size(document)

    def chunk_by_fixed_size(self, document: Document) -> List[Chunk]:
        """Chia theo số lượng từ cố định"""
        text = document.content
        if text == "":
            return []
            
        chunks = []
        step = self.chunk_size - self.chunk_overlap
        start = 0
        chunk_index = 0
        
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk_content = text[start:end]
            
            chunks.append(
                self._create_chunk(
                    doc_id=document.doc_id,
                    content=chunk_content,
                    start=start,
                    end=end,
                    index=chunk_index
                )
            )
            chunk_index += 1
            if end == len(text):
                break
            start += step
            
        return chunks

    def _create_chunk(self, doc_id: str, content: str, start: int, end: int, index: int) -> Chunk:
        """Hàm tạo Chunk"""
        if content is None:
            raise ValueError("Không có nội dung để tạo chunk")
            
        unique_chunk_id = f"{doc_id}_ch_{index}"
        return Chunk(
            chunk_id = unique_chunk_id,
            doc_id = doc_id,
            content = content,
            start = start,     # Tên biến chuẩn
            end = end,         # Tên biến chuẩn
            index = index      # Tên biến chuẩn
        )