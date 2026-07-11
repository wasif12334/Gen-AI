from dotenv import load_dotenv
# from langchain_docling.loader import DoclingLoader
# load_dotenv()

# data= DoclingLoader(".\RAG\Doumenr_loader/test.txt")
# docs=data.load()
# # print(docs)

from langchain_community.document_loaders import TextLoader
data=TextLoader(".\RAG\Doumenr_loader/test.txt")
docs=data.load()

print(len(docs))