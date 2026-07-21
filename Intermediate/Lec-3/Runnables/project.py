from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel ,RunnableLambda

load_dotenv()

model=ChatMistralAI(
    model="mistral-small-2603",
    temperature=0.5
)

#here i  am making a project in which a query is delivered to llm and three deifferent ouptup will be gernated as per different pov with having 
#single llm to all 
#prompt1
TeacherRole=ChatPromptTemplate.from_template(
    "Explain the {topic} as your are the teacher with 10 + year of experince in teaching at Universty level"
    )
#prompt2
StudentRole=ChatPromptTemplate.from_template(
    "Explain the {topic} as your are the Student  at Universty level"
    )
#prompt3
SoftwareEngRole=ChatPromptTemplate.from_template(
        "Explain the {topic} as your are the SoftwareEnger with 10 + year of experince in SoftwareEnging Multinational company"
    )

parser=StrOutputParser()

#Now i will define the sequence chain in eehcih i will run parallel

TeacherChain=TeacherRole|model|parser

StudentChain=StudentRole|model|parser

SoftwareEngChain=SoftwareEngRole|model|parser

# this will run the chains/runnables parallel
# pipeline=RunnableParallel({
#     "TeacherRole":RunnableLambda(lambda  x:x['TeacherRole'])|TeacherChain,
#     "StudentRole":RunnableLambda(lambda  x:x['StudentRole'])|StudentChain,
#     "SoftwareEngRole":RunnableLambda(lambda  x:x['SoftwareEngRole'])|SoftwareEngChain,
# })
pipeline=RunnableParallel({
    "TeacherRole":TeacherChain,
    "StudentRole":StudentChain,
    "SoftwareEngRole":SoftwareEngChain,
})


# print("\n---------------------Teacher Output------------------\n")
# result=TeacherChain.invoke("Fundemantals of Generative AI ")
# print("TeacherResponse : ",result)

# print("\n---------------------Student Output------------------\n")
# result=StudentChain.invoke("Fundemantals of Generative AI ")
# print("StudentResponse : ",result)

# print("\n--------------------SoftwareEng Output------------------\n")
# result=SoftwareEngChain.invoke("Fundemantals of Generative AI ")
# print("SoftwareEngResponse : ",result)

result=pipeline.invoke({
    
        "topic":"Machine learning"
    
    
  
})
print("\n---------------------Teacher Output------------------\n")
print(result['TeacherRole'])

print("\n---------------------Student Output------------------\n")
print(result['StudentRole'])

print("\n--------------------SoftwareEng Output------------------\n")
print(result['SoftwareEngRole'])