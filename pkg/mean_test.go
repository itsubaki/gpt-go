package pkg

import (
	"fmt"

	"github.com/itsubaki/autograd/variable"
)

func ExampleMean_basic() {
	a := M{
		{1, 2, 3},
		{4, 5, 6},
	}.Var()

	result := Mean(a)
	fmt.Println(result.Data)

	// Output: [[2] [5]]
}

func ExampleMean_withZero() {
	a := M{
		{1, 0, 1},
	}.Var()

	result := Mean(a)
	fmt.Printf("%.6f\n", result.Data[0][0])

	// Output: 0.666667
}

func ExampleMean_withZeros() {
	a := M{
		{0, 0, 0},
		{0, 0, 0},
	}.Var()

	result := Mean(a)
	fmt.Println(result.Data)

	// Output: [[0] [0]]
}

func ExampleMean_withNegatives() {
	a := M{
		{-1, 2, -3},
		{4, -5, 6},
	}.Var()

	result := Mean(a)
	fmt.Printf("%.6f %.6f\n", result.Data[0][0], result.Data[1][0])

	// Output: -0.666667 1.666667
}

func ExampleMean_gradient() {
	a := M{
		{1, 2, 3},
		{4, 5, 6},
	}.Var()

	result := Mean(a)
	result.Grad = M{
		{0.1},
		{0.2},
	}.Var()

	result.Backward()
	fmt.Println(a.Grad.Data)

	// Output: [[0.03333333333333333 0.03333333333333333 0.03333333333333333] [0.06666666666666667 0.06666666666666667 0.06666666666666667]]
}

func ExampleMean_withScalarGradient() {
	a := M{
		{1, 2},
		{3, 4},
	}.Var()

	result := Mean(a)
	result.Grad = M{
		{1.0},
		{1.0},
	}.Var()

	result.Backward()
	fmt.Println(a.Grad.Data)

	// Output: [[0.5 0.5] [0.5 0.5]]
}

func ExampleMean_inComputationGraph() {
	a := M{
		{1, 3},
		{2, 4},
	}.Var()

	meanA := Mean(a)
	result := variable.Mul(meanA, M{
		{2},
		{2},
	}.Var())

	fmt.Println(result.Data)
	result.Backward()
	fmt.Println(a.Grad.Data)

	// Output:
	// [[4] [6]]
	// [[1 1] [1 1]]
}
