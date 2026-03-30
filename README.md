# Cost-Efficient RAG POC for Tax Judgements

This is a highly optimized, low-cost Retrieval-Augmented Generation (RAG) system built specifically to process and query complex legal/tax judgments.

It is designed to minimize cloud computing costs by moving the heavy vector processing to your local machine (or container) while leveraging generous free tiers for text generation.

## 🏗️ Architecture Stack
1. **API Layer**: `FastAPI`
2. **Document Processing**: `LlamaIndex` (`SimpleDirectoryReader` with `pytesseract` / `poppler` for OCR)
3. **Embeddings (Vectorization)**: `BAAI/bge-small-en-v1.5` (Runs 100% locally - **Cost: $0**)
4. **Vector Database**: `LanceDB` (Embedded local disk storage - **Cost: $0**)
5. **LLM**: `Google Gemini 1.5 Flash` (1M token context window, highly capable, **Generous Free Tier**)

## 🚀 Quickstart via Docker

### 1. Set Up Environment Variables
Create a `.env` file in the root of the project with your Gemini API key:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```
*(You can get a free API key from [Google AI Studio](https://aistudio.google.com/)).*

### 2. Build the Docker Image
```bash
docker build -t judgement-rag .
```

### 3. Run the Container
We map the `lancedb` folder to your local machine so your database persists even if the container stops!
```bash
docker run -p 8000:8000 \
  --env-file .env \
  -v $(pwd)/lancedb:/app/lancedb \
  -v $(pwd)/data:/app/data \
  judgement-rag
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

## 📈 Production Roadmap / Next Steps
* **Advanced Document Parsing:** If your judgments have highly complex tables or poor-quality stamps, replace `SimpleDirectoryReader` with `Docling` or `Marker` to generate semantic Markdown chunks before embedding.
* **Semantic Chunking:** For legal texts, configure LlamaIndex to chunk by headers/paragraphs rather than flat word counts to maintain context.
