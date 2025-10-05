package main

import (
	"fmt"

	"github.com/ardanlabs/ai-training/foundation/vector"
)

type data struct {
	Name      string
	Authority float64 // These fields are called features.
	Animal    float64
	Human     float64
	Rich      float64
	Gender    float64
}

// Vector can convert the specified data into a vector.
func (d data) Vector() []float64 {
	return []float64{
		d.Authority,
		d.Animal,
		d.Human,
		d.Rich,
		d.Gender,
	}
}

// String pretty prints an embedding to a vector representation.
func (d data) String() string {
	return fmt.Sprintf("%f", d.Vector())
}

// =============================================================================

func main() {

	// Apply the feature dataPoints to the hand crafted embeddings.
	dataPoints := []vector.Data{
		data{Name: "Horse   ", Authority: 0.0, Animal: 1.0, Human: 0.0, Rich: 0.0, Gender: +1.0},
		data{Name: "Man     ", Authority: 0.0, Animal: 0.0, Human: 1.0, Rich: 0.0, Gender: -1.0},
		data{Name: "Woman   ", Authority: 0.0, Animal: 0.0, Human: 1.0, Rich: 0.0, Gender: +1.0},
		data{Name: "King    ", Authority: 1.0, Animal: 0.0, Human: 1.0, Rich: 1.0, Gender: -1.0},
		data{Name: "Queen   ", Authority: 1.0, Animal: 0.0, Human: 1.0, Rich: 1.0, Gender: +1.0},
	}

	// -------------------------------------------------------------------------
}
