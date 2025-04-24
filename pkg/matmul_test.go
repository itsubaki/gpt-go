package pkg

import (
	"fmt"

	"github.com/itsubaki/autograd/variable"
)

func ExampleMatMul_basic2x2() {
	a := M{
		{1, 2},
		{3, 4},
	}.Var()

	b := M{
		{5, 6},
		{7, 8},
	}.Var()

	result := MatMul(a, b)
	fmt.Println(result)

	// Output:
	// variable([[19 22] [43 50]])
}

func ExampleMatMul_nonSquare() {
	a := M{
		{1, 2, 3},
		{4, 5, 6},
	}.Var()

	b := M{
		{7, 8},
		{9, 10},
		{11, 12},
	}.Var()

	result := MatMul(a, b)
	fmt.Println(result)

	// Output:
	// variable([[58 64] [139 154]])
}

func ExampleMatMul_columnVector() {
	a := M{
		{1, 2},
		{3, 4},
		{5, 6},
	}.Var()

	b := M{
		{7},
		{8},
	}.Var()

	result := MatMul(a, b)
	fmt.Println(result)

	// Output:
	// variable([[23] [53] [83]])
}

func ExampleMatMul_rowVector() {
	a := V{1, 2}.Var()

	b := M{
		{3, 4},
		{5, 6},
	}.Var()

	result := MatMul(a, b)
	fmt.Println(result)

	// Output:
	// variable([13 16])
}

func ExampleMatMul_chain() {
	a := V{1, 2}.Var()

	b := M{
		{3, 4},
		{5, 6},
	}.Var()

	c := M{
		{7},
		{8},
	}.Var()

	result := MatMul(MatMul(a, b), c)
	fmt.Println(result)

	// Output:
	// variable([219])
}

func ExampleMatMul_zeroMatrix() {
	a := M{
		{0, 0},
		{0, 0},
	}.Var()

	b := M{
		{1, 2},
		{3, 4},
	}.Var()

	result := MatMul(a, b)
	fmt.Println(result)

	// Output:
	//variable([[0 0] [0 0]])
}

func ExampleMatMul_gradient() {
	a := M{
		{1, 2},
		{3, 4},
	}.Var()

	b := M{
		{5, 6},
		{7, 8},
	}.Var()

	result := MatMul(a, b)

	result.Grad = M{
		{1, 1},
		{1, 1},
	}.Var()

	result.Backward()

	fmt.Println(a.Grad)
	fmt.Println(b.Grad)

	// Output:
	//variable([[11 15] [11 15]])
	//variable([[4 4] [6 6]])
}

// Shortcut for building readable matrices:
//
//	M{
//	  {1, 2},
//	  {3, 4},
//	}.Var()
type M [][]float64

func (m M) Var() *variable.Variable {
	return variable.NewOf(m...)
}

type V []float64

func (v V) Var() *variable.Variable {
	return variable.NewOf(v)
}
