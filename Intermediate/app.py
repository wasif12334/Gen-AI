import tempfile
from dotenv import load_dotenv

import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="📚",
    layout="wide"
)

st.title("📚 PDF RAG Chatbot")
st.write("Upload a PDF and ask questions about it.")

uploaded_file = st.file_uploader(
    "Upload PDF",
    type=["pdf"]
)

if uploaded_file:

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name

    with st.spinner("Processing PDF..."):

        loader = PyPDFLoader(pdf_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = splitter.split_documents(docs)

        clean_chunks = []

        for chunk in chunks:

            text = chunk.page_content

            text = text.encode(
                "utf-8",
                errors="ignore"
            ).decode(
                "utf-8",
                errors="ignore"
            )

            if text.strip():
                chunk.page_content = text
                clean_chunks.append(chunk)

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        vectorstore = Chroma.from_documents(
            documents=clean_chunks,
            embedding=embeddings
        )

        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 4,
                "fetch_k": 10,
                "lambda_mult": 0.5
            }
        )

        st.session_state.retriever = retriever

    st.success("PDF processed successfully!")

if "retriever" in st.session_state:

    question = st.text_input(
        "Ask a question about your PDF"
    )

    if st.button("Ask") and question:

        retriever = st.session_state.retriever

        docs = retriever.invoke(question)

        context = "\n\n".join(
            doc.page_content
            for doc in docs
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
You are a helpful AI assistant.

Use only the provided context.

If the answer is not available in the context, say:
'I could not find enough information in the provided documents.'

Context:
{context}
"""
                ),
                (
                    "human",
                    """
Question:
{question}
"""
                )
            ]
        )

        messages = prompt.format_messages(
            context=context,
            question=question
        )

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash"
        )

        response = llm.invoke(messages)

        st.subheader("Answer")
        st.write(response.content)

        with st.expander("Retrieved Context"):
            st.write(context)