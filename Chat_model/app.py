import streamlit as st
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()

model = ChatMistralAI(
    model="mistral-small-2506",
    temperature=0.7
)

st.title("MOOD BASED AI AGENTS")

choice = st.radio(
    "Tell your response :-",
    options=[1, 2, 3],
    format_func=lambda x: {1: "press 1 for angry", 2: "press 2 for sad", 3: "press 3 for Funny"}[x]
)

if choice == 1:
    mode = "you are an angry bot reply in angry mode"
    bot = "Angry"
elif choice == 2:
    mode = "you are an sad bot reply in sad mode"
    bot = "sad"
elif choice == 3:
    mode = "you are an funny bot reply in funny mode"
    bot = "funny"

# reset memory if mode changes
if "bot" not in st.session_state or st.session_state.bot != bot:
    st.session_state.bot = bot
    st.session_state.messages = [
        SystemMessage(content=mode)
    ]

st.write(f"----------Welcome to {bot} Ai Agent -------------------------- ")

# display chat history (skip system message)
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.write(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.write(msg.content)

prompt = st.chat_input("You:")

if prompt:
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.write(prompt)

    response = model.invoke(st.session_state.messages)
    st.session_state.messages.append(AIMessage(content=response.content))

    with st.chat_message("assistant"):
        st.write(response.content)