package pkg

import (
	"fmt"
	"math"
	"strings"

	"github.com/itsubaki/autograd/function"
	"github.com/itsubaki/autograd/variable"
)

func ExampleSoftmax_basic() {
	a := M{
		{1, 2, 3},
		{4, 5, 6},
	}.Var()

	result := function.Softmax(a)
	fmt.Println(result)

	// Output:
	// variable([[0.09003057317038046 0.24472847105479764 0.6652409557748218] [0.09003057317038046 0.24472847105479764 0.6652409557748218]])
}

func ExampleSoftmax_largeValues() {
	a := V{100, 100.1, 100.2}.Var()

	result := function.Softmax(a)
	fmt.Println(result)

	// Output:
	// variable([0.30060960535572756 0.33222499353334567 0.3671654011109268])
}

func ExampleSoftmax_withMasking() {
	a := variable.NewOf(
		[]float64{1, math.Inf(-1), 3},
		[]float64{math.Inf(-1), 2, 3},
		[]float64{1, 2, math.Inf(-1)},
		[]float64{1, math.Inf(-1), math.Inf(-1)},
	)
	result := function.Softmax(a)

	for _, row := range result.Data {
		values := make([]string, len(row))
		for i, val := range row {
			values[i] = fmt.Sprintf("%.6f", val)
		}
		fmt.Println(strings.Join(values, " "))
	}

	// Output:
	// 0.119203 0.000000 0.880797
	// 0.000000 0.268941 0.731059
	// 0.268941 0.731059 0.000000
	// 1.000000 0.000000 0.000000
}

func ExampleSoftmax_allMasked() {
	a := variable.NewOf(
		[]float64{0, math.Inf(-1), math.Inf(-1)},
		[]float64{1, 2, 3},
	)
	result := function.Softmax(a)
	fmt.Println(result)

	// Output:
	// variable([[1 0 0] [0.09003057317038046 0.24472847105479764 0.6652409557748218]])
}

func ExampleSoftmax_gradient() {
	a := variable.NewOf([]float64{1, 2, 3})
	result := function.Softmax(a)
	result.Grad = variable.NewOf([]float64{1, 1, 1})
	result.Backward()
	fmt.Println(a.Grad)

	b := variable.NewOf([]float64{1, 2, 3})
	resultB := function.Softmax(b)
	resultB.Grad = variable.NewOf([]float64{1, 0, 0})
	resultB.Backward()
	fmt.Println(b.Grad)

	// Output:
	// variable([1.3877787807814457e-17 2.7755575615628914e-17 1.1102230246251565e-16])
	// variable([0.08192506906499324 -0.022033044520174298 -0.05989202454481893])
}
