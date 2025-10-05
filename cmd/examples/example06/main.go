package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/ardanlabs/ai-training/foundation/mongodb"
)

var (
	url   = "http://localhost:11434/v1/embeddings"
	model = "bge-m3:latest"

	dbName     = "example06"
	colName    = "book"
	dimensions = 1024
)

func init() {
	if v := os.Getenv("LLM_SERVER"); v != "" {
		url = v
	}

	if v := os.Getenv("LLM_MODEL"); v != "" {
		model = v
	}
}

// =============================================================================

type document struct {
	ID        int       `bson:"id"`
	Text      string    `bson:"text"`
	Embedding []float64 `bson:"embedding"`
}

// =============================================================================

func main() {
	if err := run(); err != nil {
		log.Fatal(err)
	}
}

func run() error {
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	fmt.Println("\nCreating Embeddings")

	if err := createBookEmbeddings(ctx); err != nil {
		return fmt.Errorf("createBookEmbeddings: %w", err)
	}

	// -------------------------------------------------------------------------

	fmt.Println("Initializing Database")

	client, err := mongodb.Connect(ctx, "mongodb://localhost:27017", "ardan", "ardan")
	if err != nil {
		return fmt.Errorf("mongodb.Connect: %w", err)
	}

	col, err := initDB(ctx, client)
	if err != nil {
		return fmt.Errorf("initDB: %w", err)
	}

	// -------------------------------------------------------------------------

	if err := insertBookEmbeddings(ctx, col); err != nil {
		return fmt.Errorf("insertBookEmbeddings: %w", err)
	}

	fmt.Println("\nYou can now use example07 to ask questions about this content.")

	return nil
}
