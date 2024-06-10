import numpy as np
from MyLib.Exceptions import DimensionError
from MyLib.ActivactionFunction import ActivationFunction

class Layer():
    def __init__(self, input_dim:int, output_dim:int, activation_function:ActivationFunction) -> None:
        self.in_list = None
        self.vlist = None
        self.ylist = None
        self.delta = None
        self.weights = np.random.randn(output_dim, input_dim+1)
        self.activation_func = activation_function

    def forward(self, input:np.ndarray) -> np.ndarray:
        input = np.append(input, -1) # add bias
        self.in_list = input

        if input.shape[0] != self.weights.shape[1]:
            raise DimensionError("Dimension of layer with size: {} does not match the dimension of input with size: {}".format(self.weights.shape, input.shape))

        v = np.dot(self.weights, input)
        y = self.activation_func(v)

        self.vlist = v
        self.ylist = y
        return y
    
    def backward(self, pre_delta:np.ndarray) -> None:
        self.delta = pre_delta * self.activation_func.d(self.vlist)
        return np.dot(self.weights.T, self.delta)[:-1]

    def update(self, lr) -> None:
        self.weights = self.weights + lr * np.outer(self.delta, self.in_list)


class LinearModel():
    def __init__(self, input_dim:int, output_dim:int, hidden_dim_list:list[int], activation_func:ActivationFunction) -> None:
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layer_dim_list = [input_dim] + hidden_dim_list + [output_dim]
        self.num_layers = len(self.layer_dim_list) - 1

        self.layer_list = []
        for i in range(self.num_layers):
            self.layer_list.append(Layer(self.layer_dim_list[i], self.layer_dim_list[i+1], activation_func)) # fix this

    def forward(self, input:np.ndarray) -> np.ndarray:
        x = input
        for layer in self.layer_list:
            x = layer.forward(x)
        return x
    
    def backward(self, loss_func_delta:np.ndarray, lr:float) -> None:
        delta = loss_func_delta
        for layer in self.layer_list[::-1]:
            delta = layer.backward(delta)
            layer.update(lr)
    
    def predict(self, input:np.ndarray) -> np.ndarray:
        return self.forward(input)
    
    def getWeights(self):
        return [layer.weights for layer in self.layer_list]
    
    def setWeights_1d(self, weights:np.ndarray):
        for i, layer in enumerate(self.layer_list):
            layer_shape = layer.weights.shape
            layer_size = layer.weights.size
            reshaped = weights[:layer_size].reshape(layer_shape)
            weights = weights[layer_size:]
            layer.weights = reshaped

        assert len(weights) == 0 , "weights is not empty."
    