import os
import sys
import tempfile

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import streamlit as st
from utils.styles import load_css
from src.loader import DocumentLoader
from src.chunking import TextChunker         
from src.embeddings import RAGSystem   

MODEL = "gemini-2.5-flash"
TOP_K = 3

def _set_page_config():
    st.set_page_config(
        page_title="Notebook",
        page_icon=":vietnam:",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    load_css()

def _initialize_session_state():
    for k, v in {"extracted_text": "", "pdf_name": "", "chat_history": []}.items():
        st.session_state.setdefault(k, v)
    if "rag_system" not in st.session_state:
        st.session_state.rag_system = RAGSystem()

def _render_sidebar():
    with st.sidebar:
        st.markdown(
            """
            <div style="padding-top: 8px;">
                <div class="title-sidebar">Notebook</div>
            </div>
            """, unsafe_allow_html=True
        )
        st.caption("Trợ lý Hỏi đáp tài liệu với RAG")
        
        st.markdown(
            """
            <div class="system-intro-box">
                <div class="system-intro-title">🔬 Vietnamese Q&A Rag System</div>
                <div class="system-intro-desc">
                    Hệ thống hỏi đáp thông minh dựa trên tài liệu thật.
                    Hỗ trợ đa định dạng: PDF, (đang phát triển) hình ảnh, âm thanh và video.
                </div>
                <div style="margin-top:12px; display:flex; flex-wrap:wrap; gap:6px;">
                    <span style="background:rgba(255,255,255,0.15); border-radius:20px; padding:3px 10px; font-size:11px;">📄 PDF</span>
                    <span style="background:rgba(255,255,255,0.15); border-radius:20px; padding:3px 10px; font-size:11px;">🖼️ Image</span>
                    <span style="background:rgba(255,255,255,0.15); border-radius:20px; padding:3px 10px; font-size:11px;">🎵 Audio</span>
                    <span style="background:rgba(255,255,255,0.15); border-radius:20px; padding:3px 10px; font-size:11px;">🎬 Video</span>
                </div>
            </div>
            """, unsafe_allow_html=True
        )
        
        st.markdown("---")
        with st.container(border=True):
            user_model = st.selectbox("Chọn LLM Model:", options=["gemini-2.5-flash"], index=0, key="llm_model")
            st.session_state.rag_system.llm_model = user_model
            user_topk = st.slider("Số lượng Context (k):", min_value=1, max_value=10, value=3, key="k_docs",
                                  help="Số lượng đoạn văn bản liên quan nhất được lấy từ ChromaDB để đưa vào Prompt")
            st.session_state.rag_system.top_k = user_topk

        with st.expander("Hướng dẫn nhanh"):
            st.markdown(
                """
                <div class="guide-container">
                    <div class="guide-step-row"><div class="guide-step-number">1</div><div class="guide-step-text"><span class="guide-step-action">Cấu hình</span> mô hình LLM và chỉ số ngữ cảnh top-k.</div></div>
                    <div class="guide-step-row"><div class="guide-step-number">2</div><div class="guide-step-text"><span class="guide-step-action">Tải tài liệu</span> (PDF, TXT, hoặc MD) lên và đợi hệ thống xử lý băm vector.</div></div>
                    <div class="guide-step-row"><div class="guide-step-number">3</div><div class="guide-step-text"><span class="guide-step-action">Đặt câu hỏi</span> trực tiếp dựa trên nội dung tài liệu bằng khung chat.</div></div>
                    <div class="guide-step-row"><div class="guide-step-number">4</div><div class="guide-step-text"><span class="guide-step-action">Kiểm tra</span> câu trả lời được chatbot trích xuất nguồn chính xác.</div></div>
                </div>
                """, unsafe_allow_html=True
            )

        with st.expander("Pipeline"):
            st.markdown(
                """
                <div class="guide-container">
                    <div class="guide-step-row"><div class="guide-step-number">1</div><div class="guide-step-text"><span class="guide-step-action">File</span></div></div>
                    <div class="guide-step-row"><div class="guide-step-number">2</div><div class="guide-step-text"><span class="guide-step-action">Extract Text</span></div></div>
                    <div class="guide-step-row"><div class="guide-step-number">3</div><div class="guide-step-text"><span class="guide-step-action">Chunking</span></div></div>
                    <div class="guide-step-row"><div class="guide-step-number">4</div><div class="guide-step-text"><span class="guide-step-action">Embedding</span></div></div>
                    <div class="guide-step-row"><div class="guide-step-number">5</div><div class="guide-step-text"><span class="guide-step-action">ChromaDB</span></div></div>
                    <div class="guide-step-row"><div class="guide-step-number">6</div><div class="guide-step-text"><span class="guide-step-action">Retrieval</span></div></div>
                    <div class="guide-step-row"><div class="guide-step-number">7</div><div class="guide-step-text"><span class="guide-step-action">LLM Answer</span></div></div>
                </div>
                """, unsafe_allow_html=True
            )

        st.markdown("---")
        st.markdown("💡 **THÔNG TIN LÕI HỆ THỐNG**")
        status_color = "#38a169" if st.session_state.pdf_name else "#a0aec0"
        status_text = "Đã nạp tài liệu" if st.session_state.pdf_name else "Chờ nạp tài liệu"
        
        st.markdown(
            f"""
            <div style="font-size: 13px; color: #4a5568; line-height: 1.6;">
                ● <b>LLM Model:</b> <code style="color: #3182ce;">{st.session_state.rag_system.llm_model}</code><br>
                ● <b>Embedding Engine:</b> <code style="color: #3182ce;">text-embedding-004 (Cloud)</code><br>
                ● <b>Vector Database:</b> <code style="color: #3182ce;">ChromaDB (In-Memory)</code><br>
                ● <b>Top-k result:</b> <code style="color: #3182ce;">{st.session_state.rag_system.top_k}</code><br>
                ● <b>Trạng thái:</b> <span style="color: {status_color}; font-weight: bold;">● {status_text}</span><br>
            </div>
            """, unsafe_allow_html=True
        )

def _homepage():
    st.markdown(
        """
        <div style="padding-top: 8px;">
            <div class="title">Vietnamese Q&A RAG System</div>
        </div>
        """, unsafe_allow_html=True
    )
    st.caption("Ứng dụng thông minh hỏi đáp tài liệu bằng tiếng Việt")
    
    col_chatbot, col_upload = st.columns([3, 1])
    
    with col_chatbot:
        with st.container(border=True, height=500):
            for m in st.session_state.chat_history:
                with st.chat_message(m["role"]):
                    st.write(m["content"])
            
            if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "user":        
                with st.chat_message("assistant"):
                    if st.session_state.rag_system.collection is None:
                        st.warning("Vui lòng tải tài liệu lên ở cột bên phải trước khi hỏi đáp!")
                    else:
                        with st.spinner("Đang suy nghĩ..."):
                            try:
                                answer = st.session_state.rag_system.answer(question=st.session_state.chat_history[-1]["content"])
                                st.write(answer)
                                st.session_state.chat_history.append({"role": "assistant", "content": answer})
                            except Exception as e:
                                st.error(f"Lỗi khi truy xuất câu trả lời: {e}")
            
        prompt = st.chat_input("Hỏi nội dung trong tài liệu...")
        if prompt:
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            st.rerun()

    with col_upload:
        st.markdown("Tải file tài liệu lên để hỏi đáp")
        file_uploaded = st.file_uploader("Add file", type=["pdf", "html", "txt", "md"])

        if st.session_state.pdf_name and st.session_state.extracted_text:
            with st.expander("File đang sử dụng"):
                preview_text = st.session_state.extracted_text[:1000]
                st.text_area("Nội dung trích xuất", value=preview_text, height=300)
                if len(st.session_state.extracted_text) > 1000:
                    st.info("Nội dung dài hơn 1000 ký tự, chỉ đang hiển thị phần đầu để dễ xem.")
        elif not st.session_state.extracted_text:
            with st.expander("File đang sử dụng"):
                st.write("Chưa tải file/ nội dung.")
        # st.rerun()
        if file_uploaded:
            if st.session_state.pdf_name != file_uploaded.name:
                st.session_state.rag_system.collection = None
                st.session_state.pdf_name = file_uploaded.name

            if st.session_state.rag_system.collection is None:
                file_extension = os.path.splitext(file_uploaded.name)[1].lower()
                with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
                    temp_file.write(file_uploaded.read())
                    temp_file_path = temp_file.name
                try:
                    with st.spinner(f"🔄 Đang xử lý: {file_uploaded.name}..."):
                        load_document = DocumentLoader()
                        document = load_document.load(temp_file_path)
                        st.session_state.extracted_text = document.content
                        
                        chunker = TextChunker(chunk_size=512, chunk_overlap=50)
                        chunks = chunker.chunk(document)
                        
                        st.session_state.rag_system = RAGSystem(
                            llm_model=st.session_state.rag_system.llm_model, 
                            top_k=st.session_state.rag_system.top_k
                        )
                        st.session_state.rag_system.add_chunks(chunks)
                        st.session_state.pdf_name = file_uploaded.name
                        
                    st.success(f"Nạp thành công tài liệu và tách thành {len(chunks)} chunks!")
                except Exception as e:
                    st.error(f"Có lỗi xảy ra khi xử lý file: {e}")
                finally:
                    if os.path.exists(temp_file_path):
                        os.remove(temp_file_path)
            st.rerun()
               

def main():
    _set_page_config()
    _initialize_session_state()
    _render_sidebar()
    _homepage()

if __name__ == "__main__":
    main()