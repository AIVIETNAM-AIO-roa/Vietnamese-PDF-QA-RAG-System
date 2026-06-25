import sys
import os
import tempfile

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import streamlit as st

# Mở khóa các module đã import
from src.pdf_prepare import DocumentLoader
from src.chunking import TextChunker         # Xử lý chia nhỏ văn bản
from src.embeddings import OllamaRAGSystem   # Xử lý RAG và Vector Database


def _set_page_config():
    st.set_page_config(
        page_title="Notebook",
        page_icon = ":vietnam:",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    # CSS để style 3 cột như panel cố định
    st.markdown("""
        <style>
        /* ===== RESET & BASE ===== */
        html, body, .stApp {
            font-family: "Poppins", sans-serif;
            background: #ffffff;
            color: #1a1a1a;
        }
        
        header { visibility: hidden; }
        
        .main .block-container {
            max-width: 100%;
            padding: 0 !important;
        }
        
        /* ===== TOP NAV ===== */
        .top-nav {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px 24px;
            border-bottom: 1px solid #e8e8e8;
            background: white;
            position: sticky;
            top: 0;
            z-index: 999;
        }
        
        .brand-title {
            font-size: 38px;
            font-weight: 700;
            color: #142033;
            letter-spacing: -1.5px;
        }

        .brand-subtitle {
            font-size: 20px;
            color: #667085;
            margin-top: 4px;
        }
        .top-nav {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0 24px;
            background: white;
            border-bottom: none;
            height: 52px;
        }
        
        .nav-left {
            display: flex;
            align-items: center;
            gap: 16px;        
            height: 100%;
        }
        
        .nav-btn {
            background: #1a3a6b;
            color: white !important;
            padding: 0 20px;
            height: 25px;
            display: flex;
            align-items: center;
            font-size: 13px;
            font-weight: 700;
            text-decoration: none !important;
            letter-spacing: 0.5px;
            text-transform: uppercase;
            border-radius: 6px;
            transition: background 0.15s;
        }
        
        .nav-btn:hover {
            background: #142d54;
        }
        
        /* Link thường (không có box) */
        .nav-link {
            color: #333 !important;
            padding: 0 14px;
            height: 52px;
            display: flex;
            align-items: center;
            font-size: 13px;
            font-weight: 500;
            text-decoration: none !important;
            text-transform: uppercase;
            letter-spacing: 0.3px;
        }
        
        .nav-link:hover {
            color: #1a3a6b !important;
        }
        
        .brand {
            font-size: 22px;
            font-weight: 900;
            color: #1a3a6b;
            letter-spacing: 1px;
        }
        
        /* ===== SIDEBAR ===== */
        [data-testid="stSidebar"] {
            background: #f8f9fa;
            border-right: 1px solid #e8eaed;
            padding: 0;
        }
        
        [data-testid="stSidebar"] .block-container {
            padding: 24px 16px !important;
        }
        
        /* Tiêu đề Cấu hình hiện tại */
        [data-testid="stSidebar"] h3 {
            font-size: 13px !important;
            font-weight: 700 !important;
            color: #1a1a1a !important;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            margin-bottom: 16px !important;
            padding-bottom: 8px;
            border-bottom: 1px solid #e0e0e0;
        }
        
        /* Các dòng config — st.write */
        [data-testid="stSidebar"] p {
            font-size: 13px;
            color: #444;
            margin-bottom: 6px !important;
            display: flex;
            align-items: center;
            gap: 6px;
        }
        
        [data-testid="stSidebar"] code {
            background: #e3f2fd;
            color: #1a3a6b;
            border-radius: 4px;
            padding: 2px 6px;
            font-size: 12px;
            font-weight: 600;
            border: none;
        }
            
        /* Nút Xóa lịch sử */
        [data-testid="stSidebar"] .stButton button {
            width: 100%;
            background: white;
            color: #d32f2f;
            border: 1px solid #ffcdd2;
            border-radius: 8px;
            font-size: 13px;
            font-weight: 600;
            margin-top: 8px;
            margin-bottom: 16px;
        }
        
        [data-testid="stSidebar"] .stButton button:hover {
            background: #fff5f5;
            border-color: #d32f2f;
        }
        .custom-box {
            background-color: #ffffff;
            border: 1px solid #eaeaea;
            border-radius: 12px;
            padding: 16px 20px;
            margin-top: 12px;
            margin-bottom: 12px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
                
        /* ===== 2 CỘT CHÍNH ===== */
        [data-testid="column"]:nth-child(1) {
            background: linear-gradient(135deg, rgba(26, 58, 107, 0.06), rgba(37, 99, 235, 0.06)) !important;
            border-right: 1px solid #ebebeb;
            min-height: calc(100vh - 53px);
            padding: 16px 12px !important;
        }
        
        [data-testid="column"]:nth-child(2) {
            background: linear-gradient(135deg, rgba(26, 58, 107, 0.06), rgba(37, 99, 235, 0.06)) !important;
            min-height: calc(100vh - 53px);
            padding: 16px !important;
        }
        
        [data-testid="column"]:nth-child(3) {
            background: #f9f9f9;
            border-left: 1px solid #ebebeb;
            min-height: calc(100vh - 53px);
            padding: 16px 12px !important;
        }
        
        /* ===== SECTION TITLES ===== */
        .section-title {
            font-size: 13px;
            font-weight: 600;
            color: #9b9b9b;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            margin-bottom: 12px;
        }
        
        .section-text {
            font-size: 14px;
            line-height: 1.7;
            color: #4a4a4a;
        }
        
        /* ===== FILE UPLOADER ===== */
        [data-testid="stFileUploader"] {
            background: white;
            border: 1.5px dashed #d0d0d0;
            border-radius: 10px;
            padding: 16px;
        }
        
        [data-testid="stFileUploader"]:hover {
            border-color: #1a1a1a;
        }
        
        /* ===== CHAT ===== */
        [data-testid="stChatMessage"] {
            background: white;
            border: 1px solid #ebebeb;
            border-radius: 10px;
            padding: 14px;
            margin-bottom: 10px;
        }
        
        [data-testid="stVerticalBlockBorderWrapper"] {
            border-radius: 10px !important;
            border-color: #ebebeb !important;
        }
        
        /* ===== BUTTONS ===== */
        .stButton button {
            background: white;
            color: #1a1a1a;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            font-size: 13px;
            font-weight: 500;
            min-height: 36px;
            transition: all 0.15s ease;
        }
        
        .stButton button:hover {
            background: #f5f5f5;
            border-color: #c0c0c0;
            color: #1a1a1a;
        }
        
        /* ===== STUDIO ITEMS ===== */
        .studio-item {
            background: white;
            border: 1px solid #ebebeb;
            border-radius: 10px;
            padding: 12px 14px;
            margin-bottom: 8px;
            font-size: 13px;
            font-weight: 500;
            color: #1a1a1a;
            display: flex;
            align-items: center;
            gap: 10px;
            cursor: pointer;
            transition: all 0.15s ease;
        }
        
        .studio-item:hover {
            background: #f5f5f5;
            border-color: #c0c0c0;
        }
        
        /* ===== SIDEBAR BOX ===== */
        .sidebar-box {
            background: white;
            border: 1px solid #ebebeb;
            border-radius: 10px;
            padding: 14px;
            margin-bottom: 12px;
        }
        
        .sidebar-title {
            font-size: 14px;
            font-weight: 600;
            color: #1a1a1a;
            margin-bottom: 6px;
        }
        
        .sidebar-text {
            font-size: 13px;
            color: #6b6b6b;
            line-height: 1.6;
        }
        
        /* ===== FILE BADGE ===== */
        .file-badge {
            display: inline-block;
            padding: 3px 8px;
            border-radius: 5px;
            font-size: 11px;
            font-weight: 700;
            color: white;
            margin-right: 6px;
        }
        
        /* ===== INPUT ===== */
        [data-testid="stTextInput"] input {
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            font-size: 14px;
            padding: 10px 14px;
            background: white;
        }
        
        [data-testid="stTextInput"] input:focus {
            border-color: #1a1a1a;
            box-shadow: none;
        }
        
        /* ===== HEADINGS ===== */
        h1, h2, h3 {
            color: #1a1a1a !important;
            font-weight: 600 !important;
            letter-spacing: -0.3px;
        }
        
        /* ===== SCROLLBAR ===== */
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: #e0e0e0; border-radius: 3px; }
        ::-webkit-scrollbar-thumb:hover { background: #c0c0c0; }
        </style>
        """,
        unsafe_allow_html=True)


def _initialize_session_state():
    # Khởi tạo các giá trị session_state cơ bản
    for k, v in {"collection": None, "pdf_name": "", "chat_history": []}.items():
        st.session_state.setdefault(k, v)

def _render_sidebar():
    with st.sidebar:
        st.title("Cấu hình hệ thống")
        st.caption("Ứng dụng chatbot hỏi đáp tài liệu PDF")
        st.markdown("---")
        
        # Gắn trực tiếp biến vào session_state thông qua tham số `key`
        st.selectbox("Chọn LLM Model:", options=["vicuna:7b-v1.5-q5_1"], index=0, key="llm_model")
        
        st.slider("Số lượng Context (k):", min_value=1, max_value=10, value=3, key="k_docs",
                  help="Số lượng đoạn văn bản liên quan nhất được lấy từ ChromaDB để đưa vào Prompt")
        
        st.markdown("---")
        st.markdown("💡 Trạng thái hệ thống\nMọi thứ đang hoạt động ổn định.", unsafe_allow_html=True)

def _homepage():
    st.title("Vietnamese Q&A RAG System")
    st.caption("Ứng dụng thông minh hỏi đáp tài liệu bằng tiếng Việt")
    
    col_chatbot, col_upload = st.columns([3, 1])
    
    # ================= CỘT CHATBOT ================= #
    with col_chatbot:
        # Hiển thị lại lịch sử chat trước đó
        for m in st.session_state.chat_history:
            with st.chat_message(m["role"]):
                st.write(m["content"])
                
        prompt = st.chat_input("Hỏi nội dung trong tài liệu...")
        if prompt:
            # 1. Thêm câu hỏi của user vào giao diện và lịch sử
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)
                
            # 2. Tạo câu trả lời từ hệ thống RAG
            with st.chat_message("assistant"):
                if st.session_state.collection is None:
                    st.warning("Vui lòng tải tài liệu lên ở cột bên phải trước khi hỏi đáp!")
                else:
                    with st.spinner("Đang suy nghĩ..."):
                        try:
                            rag = st.session_state.collection
                            # Gọi hàm answer từ embeddings.py để truy xuất RAG
                            answer = rag.answer(
                                question=prompt, 
                                k=st.session_state.k_docs # Truyền tham số K từ cấu hình sidebar
                            )
                            st.write(answer)
                            st.session_state.chat_history.append({"role": "assistant", "content": answer})
                        except Exception as e:
                            st.error(f"Lỗi khi truy xuất câu trả lời: {e}")

    # ================= CỘT UPLOAD FILE ================= #
    with col_upload:
        st.markdown("Tải file tài liệu lên để hỏi đáp")
        file_uploaded = st.file_uploader("Add file", type=["pdf", "html", "txt"])
        
        if file_uploaded:
            file_extension = os.path.splitext(file_uploaded.name)[1].lower()
            
            # Tạo file tạm thời trên đĩa
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
                temp_file.write(file_uploaded.read())
                temp_file_path = temp_file.name

            try:
                with st.spinner(f"🔄 Đang xử lý: {file_uploaded.name}..."):
                    # Bước 1: Nạp tài liệu
                    load_document = DocumentLoader()
                    document = load_document.load(temp_file_path)
                    
                    # Bước 2: Cắt Chunking
                    chunker = TextChunker(chunk_size=512, chunk_overlap=50)
                    chunks = chunker.chunk(document)
                    
                    # Bước 3: Tạo Vector Embeddings và lưu vào ChromaDB
                    rag_system = OllamaRAGSystem(llm_model=st.session_state.llm_model)
                    rag_system.add_chunks(chunks)
                    
                    # Bước 4: Lưu đối tượng RAG vào session state
                    st.session_state.collection = rag_system
                    st.session_state.pdf_name = file_uploaded.name
                    
                st.success(f"✅ Nạp thành công tài liệu và tách thành {len(chunks)} chunks!")

            except Exception as e:
                st.error(f"❌ Có lỗi xảy ra khi xử lý file: {e}")
                
            finally:
                # Xóa file tạm
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)

def main():
    _set_page_config() # Giữ nguyên hàm này từ mã hiện tại của bạn
    _initialize_session_state()
    _render_sidebar()
    _homepage()

if __name__ == "__main__":
    main()