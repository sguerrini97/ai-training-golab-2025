package main

import (
	"context"
	"fmt"
	"log"
	"time"

	"github.com/ardanlabs/ai-training/foundation/mongodb"
	"go.mongodb.org/mongo-driver/bson"
	"go.mongodb.org/mongo-driver/mongo"
	"go.mongodb.org/mongo-driver/mongo/options"
)

const (
	dbName     = "example4"
	colName    = "book"
	dimensions = 4
)

// =============================================================================

type document struct {
	ID        int       `bson:"id"`
	Text      string    `bson:"text"`
	Embedding []float64 `bson:"embedding"`
}

type searchResult struct {
	ID        int       `bson:"id"`
	Text      string    `bson:"text"`
	Embedding []float64 `bson:"embedding"`
	Score     float64   `bson:"score"`
}

// =============================================================================

func main() {
	if err := run(); err != nil {
		log.Fatal(err)
	}
}

func run() error {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	// -------------------------------------------------------------------------

	fmt.Println("\nConnecting to MongoDB")

	client, err := mongodb.Connect(ctx, "mongodb://localhost:27017", "ardan", "ardan")
	if err != nil {
		return fmt.Errorf("mongodb.Connect: %w", err)
	}
	defer client.Disconnect(ctx)

	// -------------------------------------------------------------------------

	fmt.Println("Initializing Database")

	col, err := initDB(ctx, client)
	if err != nil {
		return fmt.Errorf("initDB: %w", err)
	}

	// -------------------------------------------------------------------------

	return nil
}

func initDB(ctx context.Context, client *mongo.Client) (*mongo.Collection, error) {
	db := client.Database(dbName)

	col, err := mongodb.CreateCollection(ctx, db, colName)
	if err != nil {
		return nil, fmt.Errorf("createCollection: %w", err)
	}

	const indexName = "vector_index"

	settings := mongodb.VectorIndexSettings{
		NumDimensions: dimensions,
		Path:          "embedding",
		Similarity:    "cosine",
	}

	if err := mongodb.CreateVectorIndex(ctx, col, indexName, settings); err != nil {
		return nil, fmt.Errorf("createVectorIndex: %w", err)
	}

	unique := true
	indexModel := mongo.IndexModel{
		Keys:    bson.D{{Key: "id", Value: 1}},
		Options: &options.IndexOptions{Unique: &unique},
	}
	col.Indexes().CreateOne(ctx, indexModel)

	// Delete any documents that might be there.
	col.DeleteOne(ctx, bson.D{{Key: "id", Value: 1}})
	col.DeleteOne(ctx, bson.D{{Key: "id", Value: 2}})

	return col, nil
}
