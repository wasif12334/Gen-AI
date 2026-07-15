#load pdf
#split into chunks 
#creating emebeing
#sting in vector db (chroma db)
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
# from sentence_transformers import SentenceTransformer

from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
load_dotenv()


#data loading
data=PyPDFLoader("./assets/deeplearnignbook.pdf")
docs=data.load()

#SPliting the data into chunks
splitter=RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200
)
chunks= splitter.split_documents(docs)
#cleaning the data 
clean_chunks = []

for chunk in chunks:
    if isinstance(chunk.page_content, str):

        text = chunk.page_content

        # Remove invalid unicode surrogates
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

#creating emebeding
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
texts = [chunk.page_content for chunk in clean_chunks[:5]]

print(type(texts))
print(type(texts[0]))

embeddings = embedding_model.embed_documents(texts)

print("Embedding test passed")
print(len(embeddings[0]))
print("Total clean chunks:", len(clean_chunks))

for i, chunk in enumerate(clean_chunks[:10]):
    print(f"\nChunk {i}")
    print("Type:", type(chunk.page_content))
    print("Content type:", type(chunk.page_content))
#     print("Preview:", repr(chunk.page_content[:100]))
# crearting vectorstore
vectorstore=Chroma.from_documents(
    documents=clean_chunks,
    embedding=embedding_model,
    persist_directory="chroma-db"
)


