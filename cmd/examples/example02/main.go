package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/ardanlabs/ai-training/foundation/client"
	"github.com/ardanlabs/ai-training/foundation/vector"
)

var (
	url   = "http://localhost:11434/v1/embeddings"
	model = "bge-m3:latest"
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

type data struct {
	Name      string
	Text      string
	Embedding []float64 // The vector where the data is embedded in space.
}

// Vector can convert the specified data into a vector.
func (d data) Vector() []float64 {
	return d.Embedding
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

	// Construct the llm client for access the model server.
	llm := client.NewLLM(url, model)

	// -------------------------------------------------------------------------

	// Old way of representing this data with our own vector data points.
	// dataPoints := []vector.Data{
	// 	data{Name: "Horse   ", Authority: 0.0, Animal: 1.0, Human: 0.0, Rich: 0.0, Gender: +1.0},
	// 	data{Name: "Man     ", Authority: 0.0, Animal: 0.0, Human: 1.0, Rich: 0.0, Gender: -1.0},
	// 	data{Name: "Woman   ", Authority: 0.0, Animal: 0.0, Human: 1.0, Rich: 0.0, Gender: +1.0},
	// 	data{Name: "King    ", Authority: 1.0, Animal: 0.0, Human: 1.0, Rich: 1.0, Gender: -1.0},
	// 	data{Name: "Queen   ", Authority: 1.0, Animal: 0.0, Human: 1.0, Rich: 1.0, Gender: +1.0},
	// }

	// Apply the feature vectors to the hand crafted data points.
	// This time you need to use words since we are using a word based model.
	dataPoints := []vector.Data{
		data{Name: "Horse   ", Text: "Animal, Female"},
		data{Name: "Man     ", Text: "Human,  Male,   Pants, Poor, Worker"},
		data{Name: "Woman   ", Text: "Human,  Female, Dress, Poor, Worker"},
		data{Name: "King    ", Text: "Human,  Male,   Pants, Rich, Ruler"},
		data{Name: "Queen   ", Text: "Human,  Female, Dress, Rich, Ruler"},
	}

	fmt.Print("\n")

	// Iterate over each data point and use the LLM to generate the vector
	// embedding related to the model.
	for i, dp := range dataPoints {
		dataPoint := dp.(data)

		vector, err := llm.EmbedText(ctx, dataPoint.Text)
		if err != nil {
			return fmt.Errorf("embedding: %w", err)
		}

		dataPoint.Embedding = vector
		dataPoints[i] = dataPoint

		fmt.Printf("Vector: Name(%s) len(%d) %v...%v\n", dataPoint.Name, len(vector), vector[0:2], vector[len(vector)-2:])
	}

	fmt.Print("\n")

	return nil
}
