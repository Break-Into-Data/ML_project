""" FastAPI app for RAG QA pipeline 
Provides three endpoints:
- To scrape a URL and store the text in the vectorstore
- To ask a question and get an answer
- To reset the vectorstore
"""
from fastapi import FastAPI
from pydantic import BaseModel

from src.rag_pipeline import RAGPipeline


app = FastAPI()
rag = RAGPipeline()


class ScrapeRequest(BaseModel):
    url: str
    
    
class AskRequest(BaseModel):
    question: str


@app.post("/scrape/")
async def scrape_url(request: ScrapeRequest):
    """Scrape a URL and store the text in the vectorstore"""
    rag.scrape_url(request.url)
    
    return {
        "ok": True,
    }


@app.post("/ask/")
async def ask_question(request: AskRequest):
    """Ask a question and get an answer"""
    response = rag.ask_question(request.question)
    
    return {
        "ok": True,
        "answer": response,
    }


@app.post("/reset/")
async def reset_vectorstore():
    """Reset the vectorstore"""
    rag.reset_vectorstore()
    
    return {
        "ok": True,
    }
