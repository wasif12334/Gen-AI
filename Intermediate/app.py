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

# -------------------------------
# Page Config
# -------------------------------

st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 PDF RAG Assistant")
st.caption("Upload a PDF and ask questions about it using Retrieval-Augmented Generation (RAG).")

# -------------------------------
# Session State
# -------------------------------

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "retriever" not in st.session_state:
    st.session_state.retriever = None

# -------------------------------
# Upload PDF
# -------------------------------

uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)

# -------------------------------
# Process PDF
# -------------------------------

if uploaded_file is not None and st.session_state.retriever is None:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name

    with st.status("Processing your PDF...", expanded=True) as status:

        st.write("📖 Reading PDF...")
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()

        st.write(f"✅ Loaded {len(docs)} pages")

        st.write("✂ Splitting into chunks...")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = splitter.split_documents(docs)

        st.write(f"✅ Created {len(chunks)} chunks")

        st.write("🧹 Cleaning text...")

        clean_chunks = []

        for chunk in chunks:

            text = chunk.page_content.encode(
                "utf-8",
                errors="ignore"
            ).decode(
                "utf-8",
                errors="ignore"
            )

            if text.strip():
                chunk.page_content = text
                clean_chunks.append(chunk)

        st.write("🧠 Generating embeddings...")

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        st.write("📦 Creating Vector Database...")

        vectorstore = Chroma.from_documents(
            documents=clean_chunks,
            embedding=embeddings
        )

        st.write("🔍 Creating Retriever...")

        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 4,
                "fetch_k": 10,
                "lambda_mult": 0.5
            }
        )

        st.session_state.retriever = retriever
        st.session_state.chat_history = []

        status.update(
            label="✅ PDF Ready! Ask your question below.",
            state="complete"
        )

# -------------------------------
# Show Previous Chat
# -------------------------------

for chat in st.session_state.chat_history:

    with st.chat_message("user"):
        st.markdown(chat["question"])

    with st.chat_message("assistant"):
        st.markdown(chat["answer"])

# -------------------------------
# Chat Input
# -------------------------------

if st.session_state.retriever:

    question = st.chat_input("Ask a question about your PDF...")

    if question:

        with st.chat_message("user"):
            st.markdown(question)

        with st.status("Generating Answer...", expanded=True) as status:

            st.write("🔍 Searching relevant chunks...")
            docs = st.session_state.retriever.invoke(question)

            st.write(f"✅ Retrieved {len(docs)} relevant chunks")

            st.write("📄 Preparing context...")

            context = "\n\n".join(
                doc.page_content
                for doc in docs
            )

            st.write("📝 Creating prompt...")

            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        """
You are a helpful AI assistant.

Answer ONLY from the provided context.

If the answer is unavailable, reply:

'I could not find enough information in the provided document.'

Context:
{context}
"""
                    ),
                    (
                        "human",
                        "{question}"
                    )
                ]
            )

            messages = prompt.format_messages(
                context=context,
                question=question
            )

            st.write("🤖 Sending context to Gemini...")

            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash-lite"
            )

            response = llm.invoke(messages)

            answer = response.content

            # Handle structured output
            if isinstance(answer, list):

                temp = []

                for part in answer:

                    if hasattr(part, "text"):
                        temp.append(part.text)

                    elif isinstance(part, dict):
                        temp.append(part.get("text", ""))

                    else:
                        temp.append(str(part))

                answer = "\n".join(temp)

            status.update(
                label="✅ Answer Ready!",
                state="complete"
            )

        with st.chat_message("assistant"):
            st.markdown(answer)

        st.session_state.chat_history.append(
            {
                "question": question,
                "answer": answer
            }
        )