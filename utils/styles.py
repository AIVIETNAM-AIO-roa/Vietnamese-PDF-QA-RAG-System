import os
import streamlit as st

def load_css(file_path="utils/styles.css"):
    """Đọc file CSS cục bộ và nhúng trực tiếp vào giao diện Streamlit"""
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            css_content = f.read()
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
    else:
        st.warning(f"Không tìm thấy file giao diện tại đường dẫn: {file_path}")