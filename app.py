import streamlit as st
from rag import index_pdf, answer

st.set_page_config(page_title="Research Paper Assistant", page_icon="📚")
st.title("📚 Research Paper Assistant")
st.caption("Upload papers, ask questions, get answers with page citations.")

if "indexed" not in st.session_state:
    st.session_state.indexed = set()

uploaded_files = st.file_uploader(
    "Upload research papers (PDF)", type="pdf", accept_multiple_files=True
)

for f in uploaded_files or []:
    if f.name not in st.session_state.indexed:
        with st.spinner(f"Processing {f.name}..."):
            pages, chunks, ocr_pages = index_pdf(f.read(), f.name)
        st.session_state.indexed.add(f.name)
        st.success(f"`{f.name}`: {pages} pages, {chunks} chunks, {ocr_pages} pages used OCR")

question = st.text_input("Ask a question about your papers")

if question:
    with st.spinner("Searching..."):
        text, sources = answer(question)
    st.subheader("Answer")
    st.write(text)
    if sources:
        st.subheader("Sources")
        for source, page in sources:
            st.write(f"- {source}, page {page}")