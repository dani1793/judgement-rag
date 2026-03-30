package main

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"os"

	"github.com/dani1793/judgement-rag/rag"
	"github.com/gin-gonic/gin"
)

func main() {
	// Initialize RAG system
	ctx := context.Background()
	ragSys, err := rag.NewRAGSystem(ctx)
	if err != nil {
		log.Fatalf("Failed to initialize RAG system: %v", err)
	}

	r := gin.Default()

	// Ensure upload dir exists
	os.MkdirAll("data", os.ModePerm)

	r.GET("/health", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{"status": "running"})
	})

	r.POST("/upload", func(c *gin.Context) {
		file, err := c.FormFile("file")
		if err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": "File is required"})
			return
		}

		filePath := fmt.Sprintf("data/%s", file.Filename)
		if err := c.SaveUploadedFile(file, filePath); err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to save file"})
			return
		}

		// Index the document
		err = ragSys.IndexDocument(ctx, filePath)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": fmt.Sprintf("Failed to index: %v", err)})
			return
		}

		c.JSON(http.StatusOK, gin.H{"message": fmt.Sprintf("Successfully processed and indexed: %s", file.Filename)})
	})

	type QueryRequest struct {
		Query string `json:"query" binding:"required"`
	}

	r.POST("/query", func(c *gin.Context) {
		var req QueryRequest
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}

		answer, err := ragSys.Query(ctx, req.Query)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": fmt.Sprintf("Failed to query: %v", err)})
			return
		}

		c.JSON(http.StatusOK, gin.H{
			"query":  req.Query,
			"answer": answer,
		})
	})

	log.Println("Starting server on :8000")
	r.Run(":8000")
}
