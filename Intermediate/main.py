from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

#embdeing model
embeding_model=HuggingFaceEmbeddings(
      model_name="sentence-transformers/all-MiniLM-L6-v2"
)

#vectore store loaded 
vectorstore=Chroma(
    persist_directory="chroma-db",
    embedding_function=embeding_model
)
#retriver that retrive the information from the vectorestore using mmr search algo 
retriver=vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
       
        "k":4,
        "fetch-k":10,
        "lambda_mult":0.5
        
        }
)

#llm instaization
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite"
)
#prompt template

template=ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a helpful AI assistant that answers questions using the provided context.

Instructions:
1. Use only the information available in the retrieved context.
2. If the answer is not present in the context, say:
   "I could not find enough information in the provided documents."
3. Do not make up facts or use external knowledge.
4. Provide clear, concise, and accurate answers.
5. When appropriate, summarize information from multiple retrieved documents.
6. If the context contains conflicting information, mention the conflict and explain it.
7. Answer in a professional and easy-to-understand manner.

     Context:{context}
"""
        ),

    ("human","""
     Query:{question}

""")
    ]
)

print("-----------------RAG BASED APP------------------------")
print("Press 0 to EXIT")
while True:
    query=input("YOU :")
    if query==0:
        break
    docs=retriver.invoke(query)

    context="\n\n".join(
        doc.page_content for doc in docs
    )
    prommpt = template.format_messages(
     context=context,
        question=query
    )
    response=llm.invoke(prommpt)
    print("\n")
    print("AI RESPONSE :")
    print(response.content)
