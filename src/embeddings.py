import os
import chromadb
from typing import List
from src.models import Chunk

from google import genai
from google.genai import types

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
    def __init__(self, collection_name: str = "rag", llm_model: str = "gemini-2.5-flash", top_k: int = 3, api_key: str = None):
        """Construction khởi tạo RAGSystem"""
        # Lấy API Key GEMINI
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Không tìm thấy GEMINI_API_KEY. Vui lòng cấu hình API Key!")
        
        # Khởi tạo Client genai
        self.client = genai.Client(api_key=self.api_key)
        
        # Khởi tạo ChromaDB lưu trữ vector
        self.collection = chromadb.Client().get_or_create_collection(collection_name)
        
        # Các cấu hình
        self.llm_model = llm_model
        self.top_k = top_k
        self.embedding_model = "gemini-embedding-001" 


    def _embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Vector hoá tài liệu để nạp vào ChromaDB"""
        all_embeddings = []
        for text in texts:
            # Chuỗi rỗng bỏ qua
            if not text or not text.strip():
                continue
            try:
                embedding = self.client.models.embed_content(
                    model=self.embedding_model,
                    contents=str(text), 
                    config=types.EmbedContentConfig(
                        task_type="RETRIEVAL_DOCUMENT"
                    )
                )
                all_embeddings.append(embedding.embeddings[0].values)
            except Exception as e:
                print(f"Lỗi khi gọi Gemini Embedding Tài liệu: {e}")
        return all_embeddings
    

    def _embed_query(self, query: str) -> List[float]:
        """Vector hoá câu hỏi của người dùng để truy vấn"""
        try:
            response = self.client.models.embed_content(
                model=self.embedding_model,
                contents=query, 
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_QUERY"
                )
            )
            return response.embeddings[0].values
        except Exception as e:
            print(f"Lỗi khi gọi Gemini Embedding Câu hỏi: {e}")
            return []

    def add_chunks(self, chunks: List[Chunk]):
        """Dùng ChromaDB lưu trữ vector database"""
        if not chunks:
            raise ValueError("Danh sách chunks trống")
        
        ids = [str(i) for i in range(len(chunks))]
        texts = [chunk.content for chunk in chunks]
        embeddings = self._embed_documents(texts)
        
        if not embeddings:
            raise RuntimeError("Quá trình băm dữ liệu thất bại, không thể nạp vào ChromaDB.")
        
        self.collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings
        )

    def retrieve(self, query: str):
        """Truy vấn top-k vector gần nhất với query"""
        query_vector = self._embed_query(query)
        if not query_vector:
            return []
            
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=self.top_k
        )
        return results["documents"][0]

    def answer(self, question: str):
        """Đưa Prompt(Context+Question) vào LLM để tạo câu trả lời"""
        retrieved_docs = self.retrieve(question)
        if not retrieved_docs:
            return "Không tìm thấy ngữ cảnh phù hợp trong tài liệu."
            
        context = "\n\n".join(retrieved_docs)
        full_prompt = self.PROMPT_TEMPLATE.format(context=context, question=question)
        
        response = self.client.models.generate_content(
            model=self.llm_model,
            contents=full_prompt,
            config=types.GenerateContentConfig(
                temperature=0.0
            )
        )
        return response.text