from dotenv import load_dotenv
load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel ,RunnableLambda
short_prompt=ChatPromptTemplate.from_template(
"Explain the {topic} in simple words in 1-2 line"
)
detailed_prompt=ChatPromptTemplate.from_template(
"Explain the {topic} in simple words by giving real life examples in detail"
)
model=ChatMistralAI(
    model="mistral-small-2603"
)

parser=StrOutputParser()

Short_chain=detailed_prompt|model|parser
detailed_chain=detailed_prompt|model|parser
piplenine=RunnableParallel({
  "short":RunnableLambda( lambda x:x['short'])| Short_chain,
  "detailed":RunnableLambda( lambda x:x['detailed'])| detailed_chain

}
)
result=piplenine.invoke({
    "short":{
        "topic":"Machine learning"
    },
    "detailed":{
        "topic":"deep learning"
    },
  
})
print("--------------------short Prompt Output-------------------------")
print(result['short'])
print("--------------------Detailed Prompt Output-------------------------")
print(result['detailed'])