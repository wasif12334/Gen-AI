from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter 

data=PyPDFLoader(".\RAG/Doumenr_loader/Business_and_Finance.pdf")
docs=data.load()

splitter=RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=1,
)
chunks=splitter.split_documents(docs)
print(chunks[1].page_content)