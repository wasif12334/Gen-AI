from dotenv import load_dotenv
load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
# from langchain.chains import LL
prompt=ChatPromptTemplate.from_template(
"Explain the {topic} in simple words by giving real life examples"
)

model=ChatMistralAI(
    model="mistral-small-2603"
)

parser=StrOutputParser()

# formated_promt=prompt.format_messages(topic="machine learning")

# response=model.invoke(formated_promt)

# output=parser.parse(response.content)

# print(output)
#this is sequence runnable her we have to add the things to chain in sequence , PROMPT->MODEL->PARSER
chain=prompt | model | parser

result=chain.invoke("Machine Learning")

print(result)