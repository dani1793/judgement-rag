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
	"time"

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

func (r *RAGSystem) backoffRetry(ctx context.Context, op func() error, maxRetries int) error {
	var err error
	for attempt := 0; attempt <= maxRetries; attempt++ {
		if err = op(); err == nil {
			return nil
		}
		
		// If it's the last attempt, don't sleep, just return the error
		if attempt == maxRetries {
			break
		}
		
		// Backoff: Base 2 seconds * (2 ^ attempt), capped at 60 seconds
		waitSecs := 2 * (1 << uint(attempt))
		if waitSecs > 60 {
			waitSecs = 60
		}
		wait := time.Duration(waitSecs) * time.Second
		
		log.Printf("API error: %v. Retrying in %v (Attempt %d/%d)...", err, wait, attempt+1, maxRetries)
		
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(wait):
		}
	}
	return err
}

func NewRAGSystem(ctx context.Context) (*RAGSystem, error) {
	apiKey := os.Getenv("GEMINI_API_KEY")
	if apiKey == "" {
		log.Println("WARNING: GEMINI_API_KEY environment variable is not set.")
	}

	client, err := genai.NewClient(ctx, option.WithAPIKey(apiKey))
	if err != nil {
		if apiKey != "" {
			return nil, fmt.Errorf("failed to create GenAI client: %v", err)
		}
	}

	var genModel *genai.GenerativeModel
	var embedModel *genai.EmbeddingModel

	if client != nil {
		genModel = client.GenerativeModel("gemini-pro-latest")
		embedModel = client.EmbeddingModel("gemini-embedding-001")
		embedModel.TaskType = genai.TaskTypeRetrievalDocument
	}

	dbPath := "./chromem"
	db, err := chromem.NewPersistentDB(dbPath, false)
	if err != nil {
		return nil, fmt.Errorf("failed to create chromem DB: %v", err)
	}

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
		return fmt.Errorf("unsupported file format %s. Only PDF and TXT supported.", ext)
	}

	if err != nil {
		return fmt.Errorf("failed to extract text: %v", err)
	}
	if text == "" {
		return errors.New("extracted text is empty")
	}

	chunks := chunkText(text, 1000)

	if r.EmbedModel == nil {
		return errors.New("embed model not initialized (check API key)")
	}

	log.Printf("Embedding %d chunks...\n", len(chunks))
	batch := r.EmbedModel.NewBatch()
	for _, chunk := range chunks {
		batch.AddContent(genai.Text(chunk))
	}

	// === RETRY WRAPPER FOR EMBEDDING ===
	var resp *genai.BatchEmbedContentsResponse
	err = r.backoffRetry(ctx, func() error {
		var e error
		resp, e = r.EmbedModel.BatchEmbedContents(ctx, batch)
		return e
	}, 4)
	if err != nil {
		return fmt.Errorf("failed to embed chunks after backoff retries: %v", err)
	}

	var documents []chromem.Document
	for i, chunk := range chunks {
		docID := fmt.Sprintf("%s_chunk_%d", filepath.Base(filePath), i)
		metadata := map[string]string{"source": filePath}
		emb := resp.Embeddings[i].Values

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

	err = r.Collection.AddDocuments(ctx, documents, 4)
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

	// === RETRY WRAPPER FOR QUERY EMBEDDING ===
	var resp *genai.EmbedContentResponse
	err := r.backoffRetry(ctx, func() error {
		var e error
		resp, e = r.EmbedModel.EmbedContent(ctx, genai.Text(query))
		return e
	}, 4)
	if err != nil || len(resp.Embedding.Values) == 0 {
		return "", fmt.Errorf("failed to embed query after retries: %v", err)
	}

	var vec []float32
	for _, v := range resp.Embedding.Values {
		vec = append(vec, v)
	}

	resDocs, err := r.Collection.QueryEmbedding(ctx, vec, 1, nil, nil)
	if err != nil {
		return "", fmt.Errorf("failed to query collection: %v", err)
	}

	if len(resDocs) == 0 {
		return "No relevant context found in the database. Please upload a judgement first.", nil
	}

	var contextBuilder strings.Builder
	contextBuilder.WriteString("Context from Tax Judgements:\n")
	for i, doc := range resDocs {
		contextBuilder.WriteString(fmt.Sprintf("---\nChunk %d (Source: %s):\n%s\n", i+1, doc.Metadata["source"], doc.Content))
	}

	prompt := fmt.Sprintf(`Based on the following extracted legal context, please answer the query. If the context does not provide the answer, say so. 
	
%s

Query: %s`, contextBuilder.String(), query)

	// === RETRY WRAPPER FOR GENERATION ===
	var genResp *genai.GenerateContentResponse
	err = r.backoffRetry(ctx, func() error {
		var e error
		genResp, e = r.GenModel.GenerateContent(ctx, genai.Text(prompt))
		return e
	}, 4)
	if err != nil {
		return "", fmt.Errorf("failed to generate answer after retries: %v", err)
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
