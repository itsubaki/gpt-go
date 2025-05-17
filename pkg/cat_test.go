package pkg

import "fmt"

func ExampleCat_basic() {
	a := M{
		{1, 2},
		{3, 4},
	}.Var()

	b := M{
		{5, 6},
		{7, 8},
	}.Var()

	result := Cat(a, b)
	fmt.Println(result.Data)

	// Output: [[1 2 5 6] [3 4 7 8]]
}

func ExampleCat_singleInput() {
	a := M{
		{1, 2},
		{3, 4},
	}.Var()

	result := Cat(a)
	fmt.Println(result.Data)

	// Output: [[1 2] [3 4]]
}

func ExampleCat_threeMatrices() {
	a := M{
		{1, 2},
		{3, 4},
	}.Var()

	b := M{
		{5, 6},
		{7, 8},
	}.Var()

	c := M{
		{9, 10},
		{11, 12},
	}.Var()

	result := Cat(a, b, c)
	fmt.Println(result.Data)

	// Output: [[1 2 5 6 9 10] [3 4 7 8 11 12]]
}

func ExampleCat_gradient() {
	a := M{
		{1, 2},
		{3, 4},
	}.Var()

	b := M{
		{5, 6},
		{7, 8},
	}.Var()

	result := Cat(a, b)
	result.Grad = M{
		{0.1, 0.2, 0.3, 0.4},
		{0.5, 0.6, 0.7, 0.8},
	}.Var()

	result.Backward()

	fmt.Println(a.Grad.Data)
	fmt.Println(b.Grad.Data)

	// Output:
	// [[0.1 0.2] [0.5 0.6]]
	// [[0.3 0.4] [0.7 0.8]]
}

func ExampleCat_withThreeDifferentMatrices() {
	a := M{
		{1, 2},
		{3, 4},
	}.Var()

	b := M{
		{5, 6},
		{7, 8},
	}.Var()

	c := M{
		{9, 10},
		{11, 12},
	}.Var()

	result := Cat(a, b, c)
	result.Grad = M{
		{0.1, 0.2, 0.3, 0.4, 0.5, 0.6},
		{0.7, 0.8, 0.9, 1.0, 1.1, 1.2},
	}.Var()

	result.Backward()

	fmt.Println(a.Grad.Data)
	fmt.Println(b.Grad.Data)
	fmt.Println(c.Grad.Data)

	// Output:
	// [[0.1 0.2] [0.7 0.8]]
	// [[0.3 0.4] [0.9 1]]
	// [[0.5 0.6] [1.1 1.2]]
}

func ExampleCat_matrixMultiplicationWith() {
	a := M{
		{1, 2},
		{3, 4},
	}.Var()

	b := M{
		{5, 6},
		{7, 8},
	}.Var()

	c := Cat(a, b)

	d := M{
		{0.1},
		{0.2},
		{0.3},
		{0.4},
	}.Var()

	result := MatMul(c, d)
	fmt.Println(result.Data)

	result.Backward()

	fmt.Println(a.Grad.Data)
	fmt.Println(b.Grad.Data)

	// Output:
	// [[4.4] [6.4]]
	// [[0.1 0.2] [0.1 0.2]]
	// [[0.3 0.4] [0.3 0.4]]
}
