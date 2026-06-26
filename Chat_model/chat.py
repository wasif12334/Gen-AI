from dotenv import load_dotenv
load_dotenv()

#this is this the code invoking the model through chat_model and we have suceesfully run this :
# from langchain.chat_models import init_chat_model



# # model = init_chat_model("google_genai:gemini-3.5-flash")
# # print(model)
# response=model.invoke("what is AI") 
# print(response.content)


# In this we are using model though model class
# from langchain_google_genai import ChatGoogleGenerativeAI

# model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")

# response=model.invoke("who invented the motorcyle and tell me about the historty of bikes")
# print(response.content)


# now using the groq oi/model
# from langchain_groq import ChatGroq

# model=ChatGroq(model="qwen/qwen3-32b")

# response=model.invoke("who has the highest basket player in respoect to hieght")


# print(response.content)
# Now using the mistrial
from langchain_groq import ChatGroq

model=ChatGroq(model="qwen/qwen3-32b",temperature=0,max_tokens=200)

response=model.invoke("who has the highest basket player in respoect to hieght")


print(response.content)