from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel ,RunnablePassthrough

load_dotenv()

model=ChatMistralAI(
   model="mistral-small-2603",
   temperature=0.5
)

CodePromptTemplate=ChatPromptTemplate.from_messages(
    [
        ('system',"YOu are acode genertaor "),
        ('human',"{topic}")
    ]
)
ExplainCodeTemplate=ChatPromptTemplate.from_messages(
     [
        ('system',"YOu are a helpful Assistant tha explain the code in siple terms "),
        ('human',"Explain the following code in simple words\n{code}")
    ] 
)
parser=StrOutputParser()

CodeGernate_Sequence =CodePromptTemplate| model | parser 


explaincode_seq=RunnableParallel(
  { "CODE":RunnablePassthrough(),
   "EXPLAIN": ExplainCodeTemplate  | model   | parser}
)
pipeline=CodeGernate_Sequence |explaincode_seq

result=pipeline.invoke({"topic":"Write a coide plaindrome in python"})
print(result['CODE'])
print(result['EXPLAIN'])

