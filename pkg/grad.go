package pkg

func (t *Tensor) Backward() {
	// Do we need other than Ones-filled initial grad?
	t.Grad = Ones(t.Shape...)

	list := make([]*Tensor, 0)
	visited := make(map[*Tensor]bool)
	// Build topological order
	var topo func(node *Tensor)
	topo = func(node *Tensor) {
		if visited[node] {
			return
		}
		visited[node] = true

		for _, parent := range node.parents {
			topo(parent)
		}
		list = append(list, node)
	}
	topo(t)

	for _, node := range list {
		node.backward()
	}
}
