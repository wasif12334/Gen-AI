
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
load_dotenv()
embeddings = GoogleGenerativeAIEmbeddings(
    
    model="gemini-embedding-2-preview",
    output_dimensionality=7,
    
    )
documents = [
    "PAF-IAST (Pak-Austria Fachhochschule Institute of Applied Sciences and Technology). Overview: PAF-IAST is a public sector degree-awarding institute located in Haripur, Khyber Pakhtunkhwa, Pakistan..."
] 
# vector = embeddings.embed_query("hello, world!")
vector=embeddings.embed_documents(documents)
print(vector)
