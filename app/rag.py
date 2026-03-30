import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Settings
from llama_index.vector_stores.lancedb import LanceDBVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.gemini import Gemini

class RAGSystem:
    def __init__(self):
        # 1. SET UP LOCAL OPEN-SOURCE EMBEDDING MODEL (Cost: $0)
        # BAAI/bge-small-en-v1.5 is highly ranked for retrieval tasks and runs completely locally.
        print("Loading local embeddings model...")
        Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        
        # 2. SET UP GENERATION LLM (Cost: Free tier via Google API)
        # Using Gemini 1.5 Flash - Massive context window, very fast.
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("WARNING: GEMINI_API_KEY environment variable is not set. Querying will fail.")
        else:
            Settings.llm = Gemini(model="models/gemini-1.5-flash", api_key=api_key)
        
        # 3. SET UP EMBEDDED VECTOR DATABASE (Cost: $0)
        # LanceDB stores data locally on disk, avoiding expensive cloud vector DBs like Pinecone.
        print("Initializing LanceDB vector store...")
        self.vector_store = LanceDBVectorStore(uri="./lancedb", table_name="tax_judgements")
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)

        # 4. LOAD EXISTING INDEX (if any)
        try:
            self.index = VectorStoreIndex.from_vector_store(self.vector_store)
        except Exception:
            self.index = None

    def index_document(self, file_path: str):
        print(f"Indexing document: {file_path}")
        # Note: For production with complex image-heavy PDFs, swap SimpleDirectoryReader with Docling
        # SimpleDirectoryReader automatically uses PyMuPDF and pytesseract if installed.
        documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
        
        if self.index is None:
            self.index = VectorStoreIndex.from_documents(documents, storage_context=self.storage_context)
        else:
            for doc in documents:
                self.index.insert(doc)
        print("Document indexed successfully.")

    def query(self, text: str):
        if not self.index:
            return "No documents have been indexed yet. Please upload a judgement first."
        
        query_engine = self.index.as_query_engine()
        response = query_engine.query(text)
        return response
