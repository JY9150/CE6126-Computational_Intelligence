from abc import ABC, abstractmethod
from typing import Any
import numpy as np

class ActivationFunction(ABC):
    @abstractmethod
    def __call__(self, x:np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def d(self, x:np.ndarray) -> np.ndarray:
        pass

    # @abstractmethod
    # def backward(self, x:np.ndarray) -> np.ndarray:
    #     pass

class ReLu(ActivationFunction):
    def __init__(self):
        pass

    def __call__(self, x:np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def d(self, x:np.ndarray) -> np.ndarray:
        return (x > 0).astype(np.float32)


class Sigmoid():
    def __init__(self):
        pass
    
    def __call__(self, x:np.ndarray) -> np.ndarray:
        return 1/(1+np.exp(-x))
    
    def d(self, x:np.ndarray) -> np.ndarray:
        return self(x) * (1 - self(x))
    