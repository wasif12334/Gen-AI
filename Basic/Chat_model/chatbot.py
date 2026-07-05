from langchain_mistralai import ChatMistralAI
from langchain_core.messages import SystemMessage ,AIMessage,HumanMessage
from dotenv import load_dotenv
load_dotenv()

model = ChatMistralAI(
        model="mistral-small-2506",
        temperature=0.7
        

)
print("Choose your AI Mode")
print("press 1 for angry")
print("press 2 for sad")
print("press 3 for Funny ")
choice=int(input("Tell your response :- "))
if choice==1:
    mode="you are an angry bot reply in angry mode"
    bot="Angry"
elif choice==2:
    mode="you are an sad bot reply in sad mode"
    bot="sad"
elif choice==3:
    mode="you are an funny bot reply in funny mode"
    bot="funny"
# short term memory
message=[ 
    SystemMessage(
        content=mode,
    )
]
print(f"----------Welcome to {bot} Ai Agent -------------------------- ")
print("----------enter 0 toexit the chat  -------------------------- ")
while(True):

  

    prompt=input("You:")
    message.append(HumanMessage(content=prompt))
    if prompt== "0":
        break;
    
    response =model.invoke(message)
    message.append(AIMessage(content=response.content))
    print("BOT",response.content)

print(message)

   