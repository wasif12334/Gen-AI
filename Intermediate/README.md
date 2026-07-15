# RAG PDF Chatbot 🤖

This repository contains my learning and implementation of a Retrieval-Augmented Generation (RAG) chatbot using LangChain, ChromaDB, Hugging Face Embeddings, and Google Gemini.

As part of my Agentic AI learning journey, I explored how modern AI applications retrieve information from documents before generating responses.

## What I Learned

- Document Loading
- Text Splitting & Chunking
- Creating Embeddings
- ChromaDB Vector Database
- Similarity Search
- MMR Retrieval
- Multi-Query Retrieval
- Building a PDF Question-Answering Chatbot
- LangChain Fundamentals

## Tech Stack

- Python
- LangChain
- Google Gemini
- Hugging Face Embeddings
- ChromaDB

## Project Flow

```text
Document → Text Chunks → Embeddings → ChromaDB → Retriever → Gemini → Response
```

## Run the Project

### Clone Repository

```bash
git clone https://github.com/wasif12334/Gen-AI.git
cd Gen-AI/Intermediate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Add Environment Variables

Create a `.env` file and add:

```env
GOOGLE_API_KEY=your_api_key
```

### Create Vector Database

```bash
python CREATING_DB.py
```

### Run Chatbot

```bash
python main.py
```

## Current Progress

✅ Document Loaders

✅ Text Splitters

✅ Embeddings

✅ ChromaDB

✅ MMR Retrieval

✅ Multi-Query Retrieval

✅ RAG Chatbot

## What's Next?

I am currently learning and building:

- Agentic AI Systems
- LangGraph
- AI Agents
- Advanced RAG Applications
- FastAPI Deployment

## About Me

I'm Muhammad Wasif, a Software Engineering student passionate about AI, GenAI, and building real-world intelligent applications.

GitHub: https://github.com/wasif12334

⭐ If you like the project, feel free to star the repository.