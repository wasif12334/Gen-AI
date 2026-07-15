from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
load_dotenv()
#MODEL INTILIAZATION
model=ChatGoogleGenerativeAI(

    model="gemini-3.5-flash",
    temperature=0.5
)
#PDF LOADER
book=PyPDFLoader('./assets/deeplearnignbook.pdf').load()
#SPLITTING THE PDF 
splitter=RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200
)
#IN THIS HTERE ARE DEVIDED IN CHHUNKNG
chunk=splitter.split_documents(book)

#PROMPT TEMPLATE FOR MODEL
template=ChatPromptTemplate.from_messages(
    [
        ('system',"""ou are an expert AI text summarization assistant.

Your task is to read the user's input and produce a clear, accurate, and concise summary while preserving the original meaning and key information.

Guidelines:
- Identify the main ideas, important facts, and essential details.
- Remove repetition, filler words, and unnecessary information.
- Do not introduce new information, opinions, or assumptions.
- Maintain a neutral and objective tone.
- Keep names, numbers, dates, and technical terms whenever they are important.
- If the input contains multiple topics, organize the summary into logical sections or bullet points.
- Preserve the chronological order when it is important for understanding.
- If the text is already concise, provide a brief refined version instead of shortening it excessively.
- If the input is ambiguous or incomplete, summarize only the information that is explicitly provided.
- Return only the summary unless the user explicitly requests additional analysis or explanation.

Goal:
Create a summary that can be understood in less than one-third of the time required to read the original text, while retaining all critical information."""),
        ('human',"""
{data}
         """)
    ]
)
#LENGTH OF THE BOOK
print(len(book))
print("--------------------------------------------")
print("--------------------------------------------")
print("--------------------------------------------")
print("--------------------------------------------")
print("--------------------------------------------")
# FUNAL PROMPT
finalPrompt=template.format_messages(data=chunk)
repsonse=model.invoke(finalPrompt)
print(repsonse.content)