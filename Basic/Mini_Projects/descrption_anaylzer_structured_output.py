from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from pydantic import BaseModel
from typing import List ,Optional
from langchain_core.output_parsers import PydanticOutputParser

load_dotenv()
#model instailized
model= ChatMistralAI(
        model="mistral-small-2506"
        )
#this is the schema for our model tha twill give the outputin this form also allow us to make our own schama through pydantaic
class Movie(BaseModel):
    tittle:str
    Relase_year:Optional[int]
    genre:List[str]
    director:Optional[str]
    cast:List[str]
    rating:Optional[float]
    summary:str
# 
parser=PydanticOutputParser(pydantic_object=Movie)

#this is the alternative of humman,ai,system message used in pervoius case with better approach here we can define the role of model to be act as who?
promt=ChatPromptTemplate.from_messages(
 [   ('system',"""

Extract Movie information through the given Paragraph
     {format_instruction}

"""),
("human", """
{description}
""")]
)

description = input("Input your  movie descrption :- ")
final_prompt=promt.invoke(
    {
        "description":description,
        'format_instruction':parser.get_format_instructions()
    }
)

response=model.invoke(final_prompt)   
Movie_data=parser.parse(response.content)   

print("BOT: ",Movie_data)