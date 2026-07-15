#load pdf
#split into chunks 
#creating emebeing
#sting in vector db (chroma db)
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
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
clean_chunks = []

for chunk in chunks:
    if isinstance(chunk.page_content, str) and chunk.page_content.strip():
        clean_chunks.append(chunk)

print("Valid chunks:", len(clean_chunks))

#creating emebeding
embedding_model=HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
embedding_model.embed_documents(docs[:5])
print("Works")
#crearting vectorstore
vectorstore=Chroma.from_documents(
    documents=clean_chunks,
    embedding=embedding_model,
    persist_directory="chroma-db"
)

