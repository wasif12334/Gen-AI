from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
load_dotenv()


docs = [
    # ---------------- AI Related (5) ----------------
    Document(
        page_content="Artificial Intelligence (AI) is the field of computer science that focuses on building systems capable of performing tasks that normally require human intelligence such as reasoning, learning, problem-solving, and decision-making.",
        metadata={"source": "AI Fundamentals Book"}
    ),
    Document(
        page_content="Machine Learning is a subset of Artificial Intelligence where computers learn patterns from data without being explicitly programmed. It is widely used in recommendation systems, fraud detection, and image recognition.",
        metadata={"source": "Machine Learning Handbook"}
    ),
    Document(
        page_content="Deep Learning is a branch of Machine Learning that uses neural networks with multiple layers to solve complex tasks like image classification, speech recognition, and natural language processing.",
        metadata={"source": "Deep Learning Guide"}
    ),
    Document(
        page_content="Convolutional Neural Networks (CNNs) are deep learning models designed for image processing tasks. They automatically learn features such as edges, textures, and shapes from images.",
        metadata={"source": "Computer Vision Notes"}
    ),
    Document(
        page_content="Large Language Models (LLMs) such as GPT and Gemini are trained on massive text datasets. They can generate text, answer questions, summarize documents, and assist in coding.",
        metadata={"source": "LLM Research Paper"}
    ),
    # ---------------- Non-AI Related (5) ----------------
    Document(
        page_content="Python is a versatile programming language used for web development, automation, data analysis, scientific computing, and software engineering.",
        metadata={"source": "Python Programming Book"}
    ),
    Document(
        page_content="Basketball is played between two teams of five players. The objective is to score points by shooting the ball through the opponent's hoop while preventing them from scoring.",
        metadata={"source": "Basketball Guide"}
    ),
    Document(
        page_content="The solar system consists of the Sun and the planets that orbit around it, including Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune.",
        metadata={"source": "Astronomy Textbook"}
    ),
    Document(
        page_content="A healthy lifestyle includes eating a balanced diet, exercising regularly, getting enough sleep, and managing stress. These habits help improve both physical and mental health.",
        metadata={"source": "Health and Wellness Book"}
    ),
    Document(
        page_content="Pakistan is located in South Asia. Its capital city is Islamabad, and it has diverse landscapes including mountains, deserts, plains, and coastal regions.",
        metadata={"source": "Geography Book"}
   ),
]

embeding_model=HuggingFaceEmbeddings()

vector_store=Chroma.from_documents(docs,embeding_model)

retriver=vector_store.as_retriever()

llm=ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)

multi_query_retriver=MultiQueryRetriever.from_llm(
    retriever=retriver,
    llm=llm
)
query="what is the agenrtic AI"

docs=multi_query_retriver.invoke(query)

for d in docs:
    print(d.page_content)
    print(d.metadata)
    print("-" * 50)