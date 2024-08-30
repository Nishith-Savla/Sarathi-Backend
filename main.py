import os

import google.generativeai as genai
from dotenv import load_dotenv
from fastapi import FastAPI

from rag_service import RAGService

_ = load_dotenv()

app = FastAPI()

API_KEY = os.getenv("GOOGLE_API_KEY")
# Configure the Gemini API
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-pro')

# Initialize RAG service with the API key
rag_service = RAGService(env_var_name="GOOGLE_API_KEY", prompt="""
Answer the question based on the provided context.
Context:
{% for doc in documents %}
   {{ doc.content }}
{% endfor %}
Question: {{ query }}
""")


@app.get("/")
async def root():
    return {"message": "Hello, World!"}


@app.get("/items/{item_id}")
async def read_item(item_id: int):
    return {"item_id": item_id}


@app.post("/generate")
async def generate_text(prompt: dict):
    prompt = prompt.get("prompt", "")
    response = model.generate_content(prompt)
    return {"generated_text": response.text}


@app.post("/rag")
async def rag_query(query: str):
    result = rag_service.query(query)
    return {"answer": result}


@app.post("/add_documents")
async def add_documents(documents: list):
    rag_service.add_documents(documents)
    return {"message": "Documents added successfully"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
