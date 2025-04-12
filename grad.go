package main

type Op interface {
	Backward(outputGrad *Tensor)
}

func (t *Tensor) Backward() {
	if t.Grad == nil {
		t.Grad = Ones(t.Shape...)
	}

	t.Creator.Backward(t.Grad)
}

type MulGrad struct {
	x0, x1 *Tensor
}

func (m *MulGrad) Backward(outputGrad *Tensor) {
	m.x0.Grad = outputGrad.Mul(m.x1)
	if m.x0.Creator != nil {
		m.x0.Creator.Backward(m.x0.Grad)
	}
	m.x1.Grad = outputGrad.Mul(m.x0)
	if m.x1.Creator != nil {
		m.x1.Creator.Backward(m.x1.Grad)
	}
}
