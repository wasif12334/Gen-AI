from langchain_huggingface import ChatHuggingFace, HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
# from langchain_community.retrievers import 
from dotenv import load_dotenv
load_dotenv()

docs = [

    Document(
        page_content="Python is a programming language",
        metadata={"source":"Python Book"}
    ),
    Document(
        page_content="LangChain helps build LLM applications,it is used to make genrative and agentic ai in today moderen area",
        metadata={"source":"LangChain Docs"}
    ),
    Document(
        page_content="we LIVE IN THE SOCIETY WHERE WE ALL MEN TO BE THE MUSLINM",
        metadata={"source":"iSLMAIC sTUDIES BOOK"}
    )
]
embedding_model=HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore=Chroma.from_documents(
    documents=docs,
    embedding=embedding_model,
    persist_directory="chroma-db"
)
#they are not runiable and we can't oinvoke them
result=vectorstore.similarity_search("What is used from  agentic ai ")
# this is used beasuce they can runaible and we can invoke that and they have multiple searching technique
resteriver=vectorstore.as_retriever()
docs=resteriver.invoke(
"explain the agentic ai "
)
# print(docs)
for d in docs:
    print(d.page_content)