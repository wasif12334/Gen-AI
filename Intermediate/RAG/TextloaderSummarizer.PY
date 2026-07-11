from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from  langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
#model instianzation
model=ChatMistralAI(
    model="mistral-small-2506"

)
#here the file path is load ed to the data variable through text loader 
data=TextLoader(".\RAG\Doumenr_loader/test.txt")
docs=data.load() #the data is load or the content in the test.txt is load 
Temp=ChatPromptTemplate( #here the chat prompt template is used to define the role of the  ai agent 
    [
        ("system",""" You are an expert AI text summarization assistant.

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
Create a summary that can be understood in less than one-third of the time required to read the original text, while retaining all critical information. """),
        ("human",""" {data}"""),
    ]
)
#this is the fuiinal prompt which is given to invoke function and we are passing there a pagecontent of the data i have
Final_promt=Temp.format_messages(data=docs[0].page_content)

response=model.invoke(Final_promt)
print(response.content)