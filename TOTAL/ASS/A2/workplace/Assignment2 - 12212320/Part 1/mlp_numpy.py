from modules import *
class MLP(object):
    def __init__(self, n_inputs, n_hidden, n_classes):
        dims = [n_inputs] + list(n_hidden) + [n_classes]
        self.layers = []
        for i in range(len(dims) - 1):
            self.layers.append(Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                self.layers.append(ReLU())
        
    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, dout):
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout
