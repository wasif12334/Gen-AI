from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
load_dotenv()
llm =HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Pro",
    temperature=1,
    # max_new_tokens=100
)

model= ChatHuggingFace(
    llm=llm
)

response=model.invoke("what is hugging face main feature")
print(response.content)