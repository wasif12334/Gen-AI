from langchain_community.document_loaders import PyPDFLoader
data=PyPDFLoader(".\RAG\Doumenr_loader/Business_and_Finance.pdf")
docs=data.load()

print(docs[2].page_content)