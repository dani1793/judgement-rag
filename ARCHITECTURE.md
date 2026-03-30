# RAG Architecture Flow

This diagram illustrates the two primary flows in the Retrieval-Augmented Generation (RAG) system: **Indexing** (Uploading a document) and **Querying** (Asking a question). 

GitHub natively renders `mermaid.js` diagrams.

## 1. Document Indexing Flow (`/upload`)
The flow of taking a raw PDF, Image, or TXT file, running Smart OCR fallbacks if necessary, and converting it into searchable vector embeddings.

```mermaid
sequenceDiagram
    autonumber
    actor Client
    participant API as Gin HTTP Server
    participant Extractor as Native PDF Parser
    participant PDFToPPM as Poppler (pdftoppm)
    participant OCR as Tesseract OCR
    participant GeminiEmbed as Gemini API (Embedding)
    participant DB as Vector DB (chromem-go)

    Client->>API: POST /upload (Upload PDF/PNG/JPG/TXT)
    API->>API: Save file to /data
    
    alt is PNG/JPG
        API->>OCR: Run Tesseract OCR directly
        OCR-->>API: Return Extracted Text
    else is PDF
        API->>Extractor: Attempt Native Text Extraction
        Extractor-->>API: Return Extracted Text
        alt Text is empty or too short (Scanned PDF)
            API->>PDFToPPM: Convert PDF pages to temporary Images
            PDFToPPM-->>API: Return Image files
            API->>OCR: Run Tesseract OCR on each Image
            OCR-->>API: Return Extracted Text
        end
    else is TXT
        API->>API: Read raw text file
    end

    API->>API: Chunk text (e.g. 1000 chars)
    API->>GeminiEmbed: Request embeddings for N chunks (with Backoff)
    GeminiEmbed-->>API: Return float32 vectors []
    API->>API: Format documents (ID, Text, Metadata, Vector)
    API->>DB: Add documents to ChromaDB
    DB-->>API: Success
    API-->>Client: 200 OK (Indexed successfully)
```

## 2. Query Generation Flow (`/query`)
The flow of asking a question, finding relevant context, and generating an LLM response.

```mermaid
sequenceDiagram
    autonumber
    actor Client
    participant API as Gin HTTP Server
    participant GeminiEmbed as Gemini API (Embedding)
    participant DB as Vector DB (chromem-go)
    participant GeminiLLM as Gemini API (Generation)

    Client->>API: POST /query {"query": "What is the tax rate?"}
    API->>GeminiEmbed: Embed query string (with Backoff)
    GeminiEmbed-->>API: Return float32 vector []
    API->>DB: Search DB using query vector (Top K=1)
    DB-->>API: Return most relevant text chunk(s)
    API->>API: Construct prompt (Relevant Chunk + User Query)
    API->>GeminiLLM: Generate Answer based on prompt (with Backoff)
    GeminiLLM-->>API: Return generated text
    API-->>Client: 200 OK {"answer": "The tax rate is 15%"}
```
