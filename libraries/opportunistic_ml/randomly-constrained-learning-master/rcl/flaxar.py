"""
Core functionality for randomly constrained learning.
"""
__docformat__ = 'numpy'
from typing import Sequence

from jax.nn import relu

from flax.linen import Module, Dense


class NN(Module):
    """Flax NN class"""
    features: Sequence[int]
    """Sequence of hidden layer dimensions"""

    def setup(self):
        """Initialise NN weights"""
        self.layers = [Dense(f) for f in self.features]

    def __call__(self, x):
        """NN forward pass"""
        x = self.layers[0](x)
        for layer in self.layers[1:]: # No last layer relu
            x = layer(relu(x))
        return x
