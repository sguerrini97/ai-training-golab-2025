package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"regexp"
	"strings"
	"time"

	"github.com/ardanlabs/ai-training/foundation/client"
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

func createBookEmbeddings(ctx context.Context) error {
	llm := client.NewLLM(url, model)

	if _, err := os.Stat("zarf/data/book.embeddings"); err == nil {
		return nil
	}

	data, err := os.ReadFile("zarf/data/book.chunks")
	if err != nil {
		return fmt.Errorf("read file: %w", err)
	}

	output, err := os.Create("zarf/data/book.embeddings")
	if err != nil {
		return fmt.Errorf("create file: %w", err)
	}
	defer output.Close()

	fmt.Print("\n")
	fmt.Print("\033[s")

	r := regexp.MustCompile(`<CHUNK>[\w\W]*?<\/CHUNK>`)
	chunks := r.FindAllString(string(data), -1)

	// Read one chunk at a time (each line) and get the vector embedding.
	for counter, chunk := range chunks {
		fmt.Print("\033[u\033[K")
		fmt.Printf("Vectorizing Data: %d of %d", counter, len(chunks))

		chunk := strings.Trim(chunk, "<CHUNK>")
		chunk = strings.Trim(chunk, "</CHUNK>")

		// YOU WILL WANT TO KNOW HOW MANY TOKENS ARE CURRENTLY IN THE CHUNK
		// SO YOU DON'T EXCEED THE NUMBER OF TOKENS THE MODEL WILL USE TO
		// CREATE THE VECTOR EMBEDDING. THE MODEL WILL TRUNCATE YOUR CHUNK IF IT
		// EXCEEDS THE NUMBER OF TOKENS IT CAN USE TO CREATE THE VECTOR
		// EMBEDDING. THERE ARE MODELS THAT ONLY VECTORIZE AS LITTLE AS 512
		// TOKENS. THERE IS A TIKTOKEN PACKAGE IN FOUNDATION TO HELP YOU WITH
		// THIS.

		vector, err := llm.EmbedText(ctx, chunk)
		if err != nil {
			return fmt.Errorf("embedding: %w", err)
		}

		doc := document{
			ID:        counter,
			Text:      chunk,
			Embedding: vector,
		}

		data, err := json.Marshal(doc)
		if err != nil {
			return fmt.Errorf("marshal: %w", err)
		}

		// Write the json document to the embeddings file.
		if _, err := output.Write(data); err != nil {
			return fmt.Errorf("write: %w", err)
		}

		// Write a crlf for easier read access.
		if _, err := output.Write([]byte{'\n'}); err != nil {
			return fmt.Errorf("write crlf: %w", err)
		}
	}

	fmt.Print("\n")

	return nil
}
