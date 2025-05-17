package pkg

import (
	"fmt"

	"github.com/itsubaki/autograd/variable"
)

func ExampleVariance_basic() {
	a := M{
		{1, 2, 3},
		{4, 5, 6},
	}.Var()

	result := Variance(a)

	// Print with higher precision to show exact values
	fmt.Printf("%.10f\n", result.Data.At(0, 0))
	fmt.Printf("%.10f\n", result.Data.At(1, 0))

	// Output:
	// 0.6666666667
	// 0.6666666667
}

func ExampleVariance_constants() {
	a := M{
		{5, 5, 5},
		{-3, -3, -3},
	}.Var()

	result := Variance(a)

	fmt.Println(result.Data.At(0, 0))
	fmt.Println(result.Data.At(1, 0))

	// Output:
	// 0
	// 0
}

func ExampleVariance_withNegatives() {
	a := M{
		{-1, 0, 1},
		{-10, 0, 10},
	}.Var()

	result := Variance(a)

	fmt.Printf("%.10f\n", result.Data.At(0, 0))
	fmt.Printf("%.10f\n", result.Data.At(1, 0))

	// Output:
	// 0.6666666667
	// 66.6666666667
}

func ExampleVariance_gradient() {
	// Values [1, 3, 5] have a mean of 3
	a := M{
		{1, 3, 5},
	}.Var()

	result := Variance(a)

	// Print variance result
	fmt.Printf("Variance: %.10f\n", result.Data.At(0, 0))

	// Set gradient to 1.0 and backpropagate
	result.Grad = M{
		{1.0},
	}.Var()

	result.Backward()

	// Print gradients with high precision
	fmt.Printf("Gradients: %.10f %.10f %.10f\n",
		a.Grad.Data.At(0, 0), a.Grad.Data.At(0, 1), a.Grad.Data.At(0, 2))

	// Output:
	// Variance: 2.6666666667
	// Gradients: -1.3333333333 0.0000000000 1.3333333333
}

func ExampleVariance_inComputationGraph() {
	// Create input with a single row
	a := M{
		{2, 4, 6},
	}.Var()

	// Calculate variance
	v := Variance(a)

	// Multiply by scalar
	k := M{
		{0.5},
	}.Var()

	result := variable.Mul(v, k)

	// Print the result
	fmt.Printf("Result: %.10f\n", result.Data.At(0, 0))

	// Backpropagate
	result.Backward()

	// Print gradients
	fmt.Printf("Gradients: %.10f %.10f %.10f\n",
		a.Grad.Data.At(0, 0), a.Grad.Data.At(0, 1), a.Grad.Data.At(0, 2))

	// Output:
	// Result: 1.3333333333
	// Gradients: -0.6666666667 0.0000000000 0.6666666667
}
