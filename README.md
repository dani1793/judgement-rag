# Golang Cost-Efficient RAG POC for Tax Judgements

This is the **Golang** version of the ultra-low-cost Retrieval-Augmented Generation (RAG) system for querying tax judgements.

It swaps out the Python ecosystem for blazing-fast Go equivalents while maintaining a near-zero cost structure via embedded vector databases and Gemini's free tier.

## 🏗️ Architecture Stack (Go Edition)
1. **API Layer**: `github.com/gin-gonic/gin` (Fast and lightweight)
2. **Document Processing**: `github.com/ledongthuc/pdf` for native Go PDF text extraction.
3. **Embeddings & LLM**: `google/generative-ai-go/genai` using Gemini's generous free tier (`text-embedding-004` & `gemini-1.5-flash`).
4. **Vector Database**: `github.com/philippgille/chromem-go` (Zero dependency, embedded in-memory vector DB with disk persistence).


## 🗺️ Architecture Diagram
A full sequence diagram illustrating the internal RAG flow (Upload ## 🚀 Quickstart via Docker Querying) can be found in [ARCHITECTURE.md](ARCHITECTURE.md).

## 🚀 Quickstart via Docker

### 1. Set Up Environment Variables
Create a `.env` file in the root of the project with your Gemini API key:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```
*(You can get a free API key from [Google AI Studio](https://aistudio.google.com/)).*

### 2. Build the Docker Image
```bash
docker build -t judgement-rag-go .
```

### 3. Run the Container
We map the `chromem` folder to your local machine so your database persists even if the container stops!
```bash
docker run -p 8000:8000 \
  --env-file .env \
  -v $(pwd)/chromem:/app/chromem \
  -v $(pwd)/data:/app/data \
  judgement-rag-go
```

## 🧪 Testing the API

Once the server is running on `http://localhost:8000`, you can test it:

### Upload a PDF Judgement
```bash
curl -X POST -F "file=@sample_judgement.pdf" http://localhost:8000/upload
```

### Ask a Question
```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"query": "What was the final verdict regarding capital gains tax in the uploaded judgement?"}' \
  http://localhost:8000/query
```

## 📈 Learning Go Notes
* **Why `chromem-go`?** It's a fantastic pure-Go drop-in replacement for complex vector stores like Milvus or Pinecone for POCs.
* **Interfaces & Structs:** Check out `rag.go` to see how Go handles initialization without classical Object-Oriented constructors.
* **Concurrency:** Notice the `ctx context.Context` passed around everywhere? That's Go's idiomatic way of handling cancellations and timeouts for API calls!
