from src.data import Document, Chunk
from typing import List

class TextChunker():
    """
    Chia nhỏ document theo fixed_size
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ):
        if chunk_size <= 0:
            raise ValueError("chunk_size phải lớn hơn 0")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap không được âm")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap phải nhỏ hơn chunk_size")
    
        self.chunk_size = chunk_size        # Số ký tự mỗi chunk
        self.chunk_overlap = chunk_overlap  # Số ký tự chồng lấp giữa chunks

    def chunk(self, document: Document) -> List[Chunk]:
        """Chia document theo chiến lược đã cấu hình"""
        if not document.content:
            return []
        
        chunks = self.chunk_by_fixed_size(document)
        return chunks

    def chunk_by_fixed_size(self, document: Document) -> List[Chunk]:
        """Cắt theo chunk_size cố định với overlap"""
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
                    start=start, end=end, 
                    index=chunk_index, 
                )) 
            chunk_index += 1 
            if end == len(text):
                break
            start += step 
        return chunks
    
    def _create_chunk(
        self, doc_id: str, content: str, start: int, end: int, index: int
    ) -> Chunk:
        """Tạo Chunk object với chunk_id duy nhất"""
        if content is None:
            raise ValueError("Không có nội dung để tạo chunk") 
        
        unique_chunk_id = f"{doc_id}_ch_{index}"

        return Chunk(
            chunk_id = unique_chunk_id,
            doc_id = doc_id,
            content = content,
            start_index = start,
            end_index = end,
            metadata={ 
                "chunk_index": index, 
                "chunk_length": len(content), 
                "strategy": self.strategy.value
            }
        )
    