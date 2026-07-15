from langchain_huggingface import ChatHuggingFace, HuggingFaceEmbeddings
from dotenv import load_dotenv
load_dotenv()
embeddings=HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
documents = [
    "PAF-IAST (Pak-Austria Fachhochschule Institute of Applied Sciences and Technology). Overview: PAF-IAST is a public sector degree-awarding institute located in Haripur, Khyber Pakhtunkhwa, Pakistan..."
] 
vector=embeddings.embed_documents(documents)
print(len(vector))