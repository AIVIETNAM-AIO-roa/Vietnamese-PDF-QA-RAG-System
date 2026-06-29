import chromadb
import streamlit as st
from google import genai
from google.genai import types
from typing import List
from src.models import Chunk
 
# ── Cấu hình model ──────────────────────────────────────────────
LLM_MODEL       = "gemini-2.5-flash"
EMBEDDING_MODEL = "gemini-embedding-001"
 
class RAGSystem:
    # Prompt mẫu để đưa vào mô hình LLM
    PROMPT_TEMPLATE = """Bạn là một chuyên gia phân tích dữ liệu và trợ lý hỏi đáp tài liệu thông minh. 
Nhiệm vụ của bạn là dựa vào phần [NGỮ CẢNH] được cung cấp dưới đây để trả lời [CÂU HỎI] từ người dùng.

---
[NGỮ CẢNH]
{context}
---

[CÂU HỎI]
{question}

---
[HƯỚNG DẪN CẤU TRÚC PHẢN HỒI]:
1. Tuyệt đối chỉ sử dụng thông tin có trong phần [NGỮ CẢNH]. Không tự ý suy diễn hoặc dùng kiến thức bên ngoài văn bản.
2. Nếu [NGỮ CẢNH] không chứa đủ thông tin để trả lời câu hỏi, hãy phản hồi chính xác: "Xin lỗi, tài liệu được cung cấp không có thông tin về vấn đề này." Tuyệt đối không bịa đặt câu trả lời.
3. Câu trả lời cần ngắn gọn, đi thẳng vào trọng tâm, viết bằng tiếng Việt chuẩn xác và có cấu trúc mạch lạc (sử dụng gạch đầu dòng nếu có nhiều ý).

[TRẢ LỜI]:"""
    def __init__(self, collection_name: str = "rag", llm_model: str = "gemini-2.5-flash", top_k: int = 3):
        """Construction khởi tạo RAGSystem"""
        self.gemini_client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
        # Khởi tạo ChromaDB lưu trữ vector
        self.chroma_client = chromadb.Client()
        self.collection = self.chroma_client.get_or_create_collection(collection_name)
        
        # Các cấu hình
        self.llm_model = LLM_MODEL or llm_model
        self.embedding_model = EMBEDDING_MODEL 
        self.top_k = top_k

    def _embed_documents(self, chunks: list[str], batch_size: int = 50) -> list:
        """Embed theo từng batch"""
        all_embeddings = []

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            response = self.gemini_client.models.embed_content(
                model=self.embedding_model,
                contents=batch,
                config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
            )
            all_embeddings.extend([e.values for e in response.embeddings])
 
        return all_embeddings
    
    def _embed_query(self, query: str) -> list[float]:
        """Embed query"""
        response = self.gemini_client.models.embed_content(
            model=self.embedding_model,
            contents=[query],
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
        )
        return response.embeddings[0].values
    
    def add_chunks(self, chunks: List[Chunk]):
        """Embed và lưu chunks vào ChromaDB."""
        if not chunks:
            raise ValueError("Danh sách chunks trống.")
 
        ids        = [str(i) for i in range(len(chunks))]
        texts      = [chunk.content for chunk in chunks]
        embeddings = self._embed_documents(texts, batch_size=100)
 
        if not embeddings:
            raise RuntimeError("Embed thất bại, không thể nạp vào ChromaDB.")
 
        self.collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
        )

    def retrieve(self, query: str):
        """Truy vấn top-k vector gần nhất với query"""
        query_vector = self._embed_query(query)
 
        if not query_vector:
            return []
 
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=self.top_k,
        )
        return results["documents"][0]

    def answer(self, question: str):
        """Đưa Prompt(Context+Question) vào LLM để tạo câu trả lời"""
        retrieved_docs = self.retrieve(question)
 
        if not retrieved_docs:
            return "Không tìm thấy ngữ cảnh phù hợp trong tài liệu."
 
        context     = "\n\n".join(retrieved_docs)
        full_prompt = self.PROMPT_TEMPLATE.format(context=context, question=question)
 
        response = self.gemini_client.models.generate_content(
            model=LLM_MODEL,
            contents=full_prompt,
            config=types.GenerateContentConfig(temperature=0.0),
        )
        return response.text