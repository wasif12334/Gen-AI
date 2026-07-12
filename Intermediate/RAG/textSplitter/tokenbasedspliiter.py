from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import TokenTextSplitter

text_splittter=TokenTextSplitter(
    chunk_size=100,
    chunk_overlap=10
)

data=TextLoader(".\RAG/textSplitter/test.txt")
chunks=text_splittter.split_documents(data)
print(chunks)