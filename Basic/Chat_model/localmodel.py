from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from dotenv import load_dotenv
load_dotenv()

llm=HuggingFacePipeline.from_model_id(

      model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs=dict(
         max_new_tokens=512,
        do_sample=False,
        repetition_penalty=1.03,
        temprature=0.5
    )
)
chat_model = ChatHuggingFace(llm=llm)
response=chat_model.invoke("who are u?")
print(response.content)