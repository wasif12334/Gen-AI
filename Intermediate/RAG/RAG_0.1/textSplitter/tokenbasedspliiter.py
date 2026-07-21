from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import TokenTextSplitter

splittter=TokenTextSplitter(
   chunk_size=100, chunk_overlap=10
)

data=PyPDFLoader(".\RAG/Doumenr_loader/Business_and_Finance.pdf")
docs=data.load()
chunks=splittter.split_documents(docs)
print(chunks)