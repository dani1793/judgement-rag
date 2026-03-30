# RAG Architecture Flow

This diagram illustrates the two primary flows in the Retrieval-Augmented Generation (RAG) system: **Indexing** (Uploading a document) and **Querying** (Asking a question). 

GitHub natively renders `mermaid.js` diagrams.

## 1. Document Indexing Flow (`/upload`)
The flow of taking a raw PDF/TXT and converting it into searchable vector embeddings.

```mermaid
sequenceDiagram
    autonumber
    actor Client
    participant API as Gin HTTP Server
    participant Extractor as Document Parser (ledongthuc/pdf)
    participant GeminiEmbed as Gemini API (Embedding)
    participant DB as Vector DB (chromem-go)

    Client->>API: POST /upload (Upload PDF/TXT)
    API->>API: Save file to /data
    API->>Extractor: Extract text
    Extractor-->>API: Return raw text string
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
