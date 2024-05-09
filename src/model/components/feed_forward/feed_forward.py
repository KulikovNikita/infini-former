import abc

import torch

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.model.utils.typing import FPTensor
from src.model.components.activation.activations import BaseActivation, DEFAULT_ACTIVATION

from src.model.components.feed_forward.base_feed_forward import BaseFeedForward, BaseFeedForwardBuilder

class FeedForwardLayers(BaseFeedForward):
    def __init__(self,
                 hidden_size: int,
                 layer_count: int = 2,
                 dropout: float = 0.3,
                 activation: BaseActivation = DEFAULT_ACTIVATION) -> None:

        layers = FeedForwardLayers.make_layers(
            dropout = dropout,
            activation = activation,
            hidden_size = hidden_size,
            layer_count = layer_count,
        )

        super().__init__(layers = layers)

        self.__dropout = dropout
        self.__activation = activation
        self.__hidden_size = hidden_size
        self.__layer_count = layer_count

    @staticmethod
    def make_layers(dropout: float,
                    hidden_size: int,
                    layer_count: int,
                    activation: BaseActivation) -> torch.nn.ModuleList:
        layers = list()
        for _ in range(layer_count):
            convolution_net = torch.nn.Conv1d(
                kernel_size = 1,
                in_channels = hidden_size,
                out_channels = hidden_size,
            )
            dropout_net = torch.nn.Dropout(p = dropout)
            layers.extend((convolution_net, dropout_net, activation))

        result = torch.nn.ModuleList(layers[:-1])
        assert len(result) == (3 * layer_count - 1)
        return result
    
    @property
    def dropout(self) -> float:
        return self.__dropout
    
    @property
    def hidden_size(self) -> int:
        return self.__hidden_size
    
    @property
    def activation(self) -> BaseActivation:
        return self.__activation
    
    @property
    def layer_count(self) -> int:
        result = super().layer_count
        assert self.__layer_count == result
        return result
    
    def _forward(self, input: FPTensor) -> FPTensor:
        _, _, input_dim = input.size()
        assert input_dim == self.hidden_size
        input_fixed = input.permute(0, 2, 1)
        output = input_fixed
        for layer in self.layers:
            output = layer(output)
        output_fixed = output.permute(0, 2, 1)
        assert output_fixed.size() == input.size()
        return output_fixed
    
class FeedForwardLayersBuilder(BaseFeedForwardBuilder):
    def __init__(self, 
                 hidden_size: int,
                 layer_count: int = 2,
                 dropout: float = 0.3,
                 activation: BaseActivation = DEFAULT_ACTIVATION) -> None:
        self.dropout = dropout
        self.activation = activation
        self.hidden_size = hidden_size
        self.layer_count = layer_count
    
    def build(self) -> FeedForwardLayers:
        return FeedForwardLayers(
            dropout = self.dropout,
            activation = self.activation,
            layer_count = self.layer_count,
            hidden_size = self.hidden_size,
        )

import logging
import unittest

class TestFeedForwardLayers(unittest.TestCase):
    def _make_generator(self, seed = 777):
        gen = torch.Generator()
        return gen.manual_seed(seed)
    
    def _dummy_batch(self, gen, **kwargs):
        size = (kwargs["batch_size"], kwargs["seq_len"], kwargs["hidden_size"])
        return torch.rand(*size, generator = gen, dtype = torch.float32)

    def assert_sizes(self, **kwargs):
        seed = 777 + sum(kwargs.values())
        gen = self._make_generator(seed = seed)
        input = self._dummy_batch(gen, **kwargs)

        network = FeedForwardLayersBuilder(
            hidden_size = kwargs["hidden_size"],
            layer_count = kwargs["layer_count"],
        ).build()
        output = network(input)

        self.assertEqual(output.size(), input.size())
        
    def test_determined_sizes(self):
        sizes: typing.List[typing.Mapping[str, int]] = [
            {"batch_size": 1, "seq_len": 1, "hidden_size": 1, "layer_count": 1},
            {"batch_size": 10, "seq_len": 1, "hidden_size": 1, "layer_count": 1},
            {"batch_size": 1, "seq_len": 10, "hidden_size": 1, "layer_count": 1},
            {"batch_size": 1, "seq_len": 1, "hidden_size": 10, "layer_count": 1},
            {"batch_size": 1, "seq_len": 1, "hidden_size": 1, "layer_count": 10},
            {"batch_size": 9, "seq_len": 10, "hidden_size": 11, "layer_count": 12},
            {"batch_size": 99, "seq_len": 111, "hidden_size": 127, "layer_count": 23},
        ]

        for s in sizes:
            self.assert_sizes(**s)

if __name__ == "__main__":
    logging.getLogger("").setLevel(logging.DEBUG)
    unittest.main()
