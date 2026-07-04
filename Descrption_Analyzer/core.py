from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
load_dotenv()

model= ChatMistralAI(
        model="mistral-small-2506"
        )

promt=ChatPromptTemplate.from_messages(
    [

    ("system","""You are an AI movie assistant.

Analyze the following movie description.

Your tasks are:
- Extract all important information.
- Identify the movie title (if available).
- Identify the genre.
- Identify the main characters.
- Identify the director.
- Identify the cast.
- Identify the setting.
- Explain the main plot in a few sentences.
- Write a short summary (around 100 words).
- Mention any other important details you find.



Be accurate, concise, and do not invent missing information."""),
("human","""
Extract information from this Movie Descrption :
 {Descrption}
""")]
)

Descrption = input("Input your  movie descrption")
final_prompt=promt.invoke(
    {
        "Descrption":Descrption
    }
)

response=model.invoke(final_prompt)   

print("BOT: ",response.content)