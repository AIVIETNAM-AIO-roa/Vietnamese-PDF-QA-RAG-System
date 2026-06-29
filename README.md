# 🔬 Vietnamese Q&A RAG System

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![Google Gemini](https://img.shields.io/badge/Gemini-2.5_Flash-4285F4?style=flat&logo=google&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-In--Memory-orange?style=flat)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat)

> **[👉 Trải nghiệm trực tiếp (Live Demo) tại đây!](https://vietnamese-pdf-app-rag-system.streamlit.app)**

Hệ thống hỏi đáp thông minh dựa trên tài liệu (**RAG – Retrieval-Augmented Generation**) được thiết kế giao diện thân thiện. Người dùng tải lên tài liệu cá nhân, đặt câu hỏi, và AI sẽ **chỉ trả lời dựa trên nội dung tài liệu**, giải quyết triệt để vấn đề "ảo giác" (hallucination) thường gặp ở các mô hình ngôn ngữ lớn.

---

## ✨ Tính năng nổi bật

| Tính năng | Chi tiết |
|---|---|
| **Hỗ trợ định dạng PDF** | Tự động nhận diện và trích xuất văn bản từ tệp PDF (qua `pypdf`) và văn bản thuần túy |
| **Chunking thông minh** | Chia văn bản thành các đoạn nhỏ với `chunk_size=512`, `chunk_overlap=50` để đảm bảo ngữ cảnh liền mạch |
| **Embedding chất lượng cao** | Sử dụng mô hình `gemini-embedding-001` (Google GenAI) với `task_type="RETRIEVAL_DOCUMENT"` |
| **Vector Database cục bộ** | Lưu trữ và truy vấn vector với **ChromaDB** (In-Memory), không cần cấu hình server |
| **LLM trả lời chính xác** | Dùng `gemini-2.5-flash` với `temperature=0.0` và prompt chuyên sâu, ép buộc AI bám sát tài liệu |
| **Tùy biến linh hoạt** | Thanh trượt `top_k` (1–10) cho phép người dùng điều chỉnh số đoạn ngữ cảnh đưa vào LLM |

---

## 📂 Cấu trúc thư mục

```text
Vietnamese-PDF-QA-RAG-System/
├── app.py                  # Điểm khởi chạy chính – giao diện Streamlit
├── requirements.txt        # Danh sách thư viện phụ thuộc
├── src/                    # Mã nguồn lõi của pipeline RAG
│   ├── loader.py           # Nạp & trích xuất nội dung (PDFLoader, TEXTLoader)
│   ├── chunking.py         # Chia nhỏ văn bản (TextChunker)
│   └── embeddings.py       # Nhúng vector, ChromaDB và tích hợp LLM (RAGSystem)
├── utils/
│   └── styles.py           # CSS tùy chỉnh cho giao diện Streamlit
└── notebooks/              # Các file thử nghiệm & prototyping (.ipynb)
```

---

## 🚀 Hướng dẫn Cài đặt & Khởi chạy

### 1. Yêu cầu hệ thống

- Python **3.9+**
- API Key hợp lệ từ [Google AI Studio](https://aistudio.google.com/) (`GEMINI_API_KEY`)

### 2. Clone dự án & cài đặt thư viện

```bash
git clone https://github.com/<your-username>/Vietnamese-PDF-QA-RAG-System.git
cd Vietnamese-PDF-QA-RAG-System
pip install -r requirements.txt
```

> Các thư viện chính bao gồm: `streamlit`, `chromadb`, `google-genai`, `pypdf`.

### 3. Cấu hình API Key

Tạo thư mục `.streamlit` ở thư mục gốc, sau đó tạo tệp `secrets.toml` bên trong:

```toml
# .streamlit/secrets.toml
GEMINI_API_KEY = "your_google_gemini_api_key_here"
```

> ⚠️ Không commit tệp `secrets.toml` lên Git. Hãy thêm `.streamlit/secrets.toml` vào `.gitignore`.

### 4. Khởi chạy ứng dụng

```bash
streamlit run app.py
```

Ứng dụng sẽ tự động mở trên trình duyệt tại: **`http://localhost:8501`**

---

## 💡 Luồng hoạt động hệ thống (Pipeline)

```
Tài liệu đầu vào
      │
      ▼
① Nạp tài liệu      ──► PDFLoader / TEXTLoader trích xuất văn bản thô
      │
      ▼
② Chunking         ──► TextChunker chia thành các đoạn chunk_size=512
      │
      ▼
③ Embedding      ──► gemini-embedding-001 chuyển hoá mỗi đoạn → vector
      │
      ▼
④ Lưu vào ChromaDB  ──► Lưu trữ In-Memory, sẵn sàng truy vấn
      │
      ▼
⑤ Câu hỏi người dùng ► Nhúng câu hỏi → tìm top-k vector gần nhất
      │
      ▼
⑥ Sinh câu trả lời  ──► Đưa ngữ cảnh + câu hỏi vào gemini-2.5-flash
      │
      ▼
   Câu trả lời chính xác, bám sát tài liệu
```

---

## 🛠️ Công nghệ sử dụng

- **[Streamlit](https://streamlit.io/)** – Giao diện web tương tác
- **[Google Generative AI](https://ai.google.dev/)** – Embedding (`gemini-embedding-001`) & LLM (`gemini-2.5-flash`)
- **[ChromaDB](https://www.trychroma.com/)** – Vector database nhẹ, chạy In-Memory
- **[pypdf](https://pypdf.readthedocs.io/)** – Trích xuất văn bản từ PDF

---

## 🗺️ Lộ trình phát triển (Roadmap)

Dự án hiện đang hỗ trợ tài liệu PDF và văn bản thuần túy. Các định dạng đầu vào sau đây đang được lên kế hoạch phát triển trong các phiên bản tiếp theo:

| Định dạng | Trạng thái | Mô tả |
|---|---|---|
| 📄 PDF / TXT | Hoàn thiện | Trích xuất và hỏi đáp trên tài liệu văn bản |
| 📝 DOCX / Markdown | Đang hoàn thiện | Mở rộng hỗ trợ các định dạng văn bản phổ biến khác |
| 🎵 Audio | Đang hoàn thiện | Phiên âm tự động (Speech-to-Text) rồi đưa vào pipeline RAG |
| 🖼️ Image/Video | Đang hoàn thiện | Trích xuất văn bản từ ảnh (OCR) và hiểu nội dung hình ảnh/ đoạn phim (Vision) |
| 🎙️ Studio | Đang nghiên cứu |Tạo nội dung học tập từ tài liệu: quiz trắc nghiệm, flashcard, tóm tắt, podcast dạng hội thoại |