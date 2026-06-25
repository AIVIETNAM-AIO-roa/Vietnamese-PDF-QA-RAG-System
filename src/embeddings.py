import ollama
import chromadb
from typing import List
from src.data import Chunk

class OllamaRAGSystem:
    PROMPT_TEMPLATE = """Bạn là trợ lý hỏi đáp. Dùng các đoạn ngữ cảnh dưới đây để trả lời câu hỏi. 
    Nếu ngữ cảnh không có thông tin, hãy nói là bạn không biết, đừng bịa! Trả lời ngắn gọn, chính xác, bằng tiếng Việt.
    Ngữ cảnh: {context}
    Câu hỏi: {question}
    Trả lời:"""

    def __init__(self, collection_name: str = "rag", llm_model: str = "vicuna:7b-v1.5-q5_1"):
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(collection_name)
        self.llm_model = llm_model

    def _embed(self, texts: List[str]):
        return ollama.embed(model="bge-m3", input=texts)["embeddings"]

    def add_chunks(self, chunks: List[Chunk]):
        if not chunks:
            raise ValueError("Danh sách chunks trống")
        
        ids = [str(i) for i in range(len(chunks))]
        # Lấy nội dung text để đưa vào ChromaDB
        texts = [chunk.content for chunk in chunks]
        
        self.collection.add(
            ids=ids,
            documents=texts,
            embeddings=self._embed(texts)
        )

    def retrieve(self, query: str, top_k: int = 3):
        results = self.collection.query(
            query_embeddings=self._embed([query]),
            n_results=top_k
        )
        return results["documents"]

    def answer(self, question: str, k: int = 4):
        # Tránh lỗi keyword nếu truy xuất sai tên
        retrieved_docs = self.retrieve(question, top_k=k)
        context = "\n\n".join(retrieved_docs)
        full_prompt = self.PROMPT_TEMPLATE.format(context=context, question=question)
        
        response = ollama.chat(
            model=self.llm_model,
            messages=[{"role": "user", "content": full_prompt}], # message -> messages
            options={"temperature": 0}
        )
        return response["message"]["content"]