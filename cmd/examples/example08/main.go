package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"time"
)

var (
	url   = "http://localhost:11434/v1/chat/completions"
	model = "gpt-oss:latest"
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

func main() {
	if err := run(); err != nil {
		log.Fatal(err)
	}
}

func run() error {
	ctx, cancel := context.WithTimeout(context.Background(), 15*60*time.Second)
	defer cancel()

	db, err := initSQLDB(ctx)
	if err != nil {
		return fmt.Errorf("initSQLDB: %w", err)
	}
	defer db.Close()

	// -------------------------------------------------------------------------

	return nil
}
