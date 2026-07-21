from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
splitter=CharacterTextSplitter(
chunk_size=100,
chunk_overlap=10,

)
data=TextLoader(".\RAG/textSplitter/test.txt")
docs=data.load()
chunks=splitter.split_documents(docs)
print(len(chunks))
for i in chunks:
    print(i.page_content)
    print("-----------------")
    print("-----------------")
    print("-----------------")
    print("-----------------")