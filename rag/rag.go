package rag

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"

	"github.com/google/generative-ai-go/genai"
	"github.com/ledongthuc/pdf"
	"github.com/philippgille/chromem-go"
	"google.golang.org/api/option"
)

type RAGSystem struct {
	GenAIClient    *genai.Client
	GenModel       *genai.GenerativeModel
	EmbedModel     *genai.EmbeddingModel
	ChromaDBClient *chromem.DB
	Collection     *chromem.Collection
}

func NewRAGSystem(ctx context.Context) (*RAGSystem, error) {
	apiKey := os.Getenv("GEMINI_API_KEY")
	if apiKey == "" {
		log.Println("WARNING: GEMINI_API_KEY environment variable is not set.")
	}

	// 1. Initialize Gemini Client for Generation & Embeddings
	// For this Go version, we use Gemini's free embedding API instead of CGO/local models for simplicity & zero cost.
	client, err := genai.NewClient(ctx, option.WithAPIKey(apiKey))
	if err != nil {
		if apiKey != "" {
			return nil, fmt.Errorf("failed to create GenAI client: %v", err)
		}
	}

	var genModel *genai.GenerativeModel
	var embedModel *genai.EmbeddingModel

	if client != nil {
		genModel = client.GenerativeModel("gemini-2.0-flash")
		embedModel = client.EmbeddingModel("gemini-embedding-001")
		embedModel.TaskType = genai.TaskTypeRetrievalDocument
	}

	// 2. Initialize Chromem-go (Embedded Vector Database, similar to LanceDB)
	dbPath := "./chromem"
	db, err := chromem.NewPersistentDB(dbPath, false)
	if err != nil {
		return nil, fmt.Errorf("failed to create chromem DB: %v", err)
	}

	// Create or get collection
	collection, err := db.GetOrCreateCollection("tax_judgements", nil, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create collection: %v", err)
	}

	return &RAGSystem{
		GenAIClient:    client,
		GenModel:       genModel,
		EmbedModel:     embedModel,
		ChromaDBClient: db,
		Collection:     collection,
	}, nil
}

// ExtractTextFromPDF uses ledongthuc/pdf to extract raw text
// For advanced OCR (PNG/JPG), you'd integrate an external exec call to tesseract.
func ExtractTextFromPDF(path string) (string, error) {
	f, r, err := pdf.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()

	var buf bytes.Buffer
	b, err := r.GetPlainText()
	if err != nil {
		return "", err
	}
	buf.ReadFrom(b)
	return buf.String(), nil
}

// chunkText simply splits the text into ~1000 character chunks for embedding
func chunkText(text string, chunkSize int) []string {
	var chunks []string
	runes := []rune(text)
	for i := 0; i < len(runes); i += chunkSize {
		end := i + chunkSize
		if end > len(runes) {
			end = len(runes)
		}
		chunks = append(chunks, string(runes[i:end]))
	}
	return chunks
}

func (r *RAGSystem) IndexDocument(ctx context.Context, filePath string) error {
	log.Printf("Indexing document: %s\n", filePath)

	ext := strings.ToLower(filepath.Ext(filePath))
	var text string
	var err error

	if ext == ".pdf" {
		text, err = ExtractTextFromPDF(filePath)
	} else if ext == ".txt" {
		b, err2 := os.ReadFile(filePath)
		text = string(b)
		err = err2
	} else {
		return fmt.Errorf("unsupported file format %s. Only PDF and TXT supported in this POC.", ext)
	}

	if err != nil {
		return fmt.Errorf("failed to extract text: %v", err)
	}

	if text == "" {
		return errors.New("extracted text is empty")
	}

	// Create chunks
	chunks := chunkText(text, 1000)

	// Embed chunks using Gemini Embedding Model
	if r.EmbedModel == nil {
		return errors.New("embed model not initialized (check API key)")
	}

	log.Printf("Embedding %d chunks...\n", len(chunks))
	batch := r.EmbedModel.NewBatch()
	for _, chunk := range chunks {
		batch.AddContent(genai.Text(chunk))
	}

	resp, err := r.EmbedModel.BatchEmbedContents(ctx, batch)
	if err != nil {
		return fmt.Errorf("failed to embed chunks: %v", err)
	}

	// Insert into Chroma
	var documents []chromem.Document
	for i, chunk := range chunks {
		docID := fmt.Sprintf("%s_chunk_%d", filepath.Base(filePath), i)
		metadata := map[string]string{"source": filePath}
		emb := resp.Embeddings[i].Values

		// Ensure float32 compatibility for chromem
		var vec []float32
		for _, v := range emb {
			vec = append(vec, v)
		}

		documents = append(documents, chromem.Document{
			ID:        docID,
			Content:   chunk,
			Metadata:  metadata,
			Embedding: vec,
		})
	}

	err = r.Collection.AddDocuments(ctx, documents, 4) // concurrency 4
	if err != nil {
		return fmt.Errorf("failed to add documents to Chroma: %v", err)
	}

	log.Println("Document indexed successfully.")
	return nil
}

func (r *RAGSystem) Query(ctx context.Context, query string) (string, error) {
	if r.Collection == nil || r.GenModel == nil || r.EmbedModel == nil {
		return "", errors.New("system not fully initialized (missing API key or collection)")
	}

	// 1. Embed the query
	resp, err := r.EmbedModel.EmbedContent(ctx, genai.Text(query))
	if err != nil || len(resp.Embedding.Values) == 0 {
		return "", fmt.Errorf("failed to embed query: %v", err)
	}

	var vec []float32
	for _, v := range resp.Embedding.Values {
		vec = append(vec, v)
	}

	// 2. Retrieve top chunks from Chroma
	resDocs, err := r.Collection.Query(ctx, query, 1, nil, nil)
	if err != nil {
		// chromem-go supports direct query by vector but its Query uses an internal embedding func if set.
		// For our manual embeddings, we use QueryEmbedding.
	}

	// Using explicit vector query instead
	resDocs, err = r.Collection.QueryEmbedding(ctx, vec, 1, nil, nil)
	if err != nil {
		return "", fmt.Errorf("failed to query collection: %v", err)
	}

	if len(resDocs) == 0 {
		return "No relevant context found in the database. Please upload a judgement first.", nil
	}

	// 3. Construct prompt
	var contextBuilder strings.Builder
	contextBuilder.WriteString("Context from Tax Judgements:\n")
	for i, doc := range resDocs {
		contextBuilder.WriteString(fmt.Sprintf("---\nChunk %d (Source: %s):\n%s\n", i+1, doc.Metadata["source"], doc.Content))
	}

	prompt := fmt.Sprintf(`Based on the following extracted legal context, please answer the query. If the context does not provide the answer, say so. 
	
%s

Query: %s`, contextBuilder.String(), query)

	// 4. Generate Answer
	genResp, err := r.GenModel.GenerateContent(ctx, genai.Text(prompt))
	if err != nil {
		return "", fmt.Errorf("failed to generate answer: %v", err)
	}

	if len(genResp.Candidates) == 0 || len(genResp.Candidates[0].Content.Parts) == 0 {
		return "No answer generated.", nil
	}

	var answer strings.Builder
	for _, part := range genResp.Candidates[0].Content.Parts {
		if text, ok := part.(genai.Text); ok {
			answer.WriteString(string(text))
		}
	}

	return answer.String(), nil
}
