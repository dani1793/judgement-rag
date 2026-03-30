import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from app.rag import RAGSystem

app = FastAPI(
    title="Cost-Efficient Tax Judgement RAG POC",
    description="Upload PDFs/Images of judgements, index them using open-source embeddings, and query them using Gemini Flash."
)

# Ensure the data and database directories exist
os.makedirs("data", exist_ok=True)
os.makedirs("lancedb", exist_ok=True)

# Initialize RAG System
rag_system = RAGSystem()

class QueryRequest(BaseModel):
    query: str

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    try:
        file_path = f"data/{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Index the document
        rag_system.index_document(file_path)
        return {"message": f"Successfully processed and indexed: {file.filename}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query_documents(request: QueryRequest):
    try:
        response = rag_system.query(request.query)
        return {
            "query": request.query,
            "answer": str(response)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "running"}
