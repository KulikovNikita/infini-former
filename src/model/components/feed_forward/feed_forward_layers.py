import torch

import typing
import dataclasses

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.model.utils.builders import BaseBuilder

from src.model.components.activation.activations import DEFAULT_ACTIVATION
from src.model.components.activation.base_activation import MaybeActivationBuilder

from src.model.components.feed_forward.base_feed_forward import BaseFeedForward
from src.model.components.feed_forward.feed_forward_network import FeedForwardNetwork
from src.model.components.feed_forward.feed_forward_block import FeedForwardBlockBuilder

class FeedForwardLayers(FeedForwardNetwork):
    def __init__(self, 
                 hidden_size: int,
                 block_count: int = 1,
                 dropout: float = 0.3,
                 activation: MaybeActivationBuilder = DEFAULT_ACTIVATION) -> None:
        builders = FeedForwardLayers.make_block_builders(
            dropout = dropout,
            activation = activation,
            hidden_size = hidden_size,
            block_count = block_count,
        )

        super().__init__(blocks = builders)

    @staticmethod
    def make_block_builders(hidden_size: int,
                            block_count: int,
                            dropout: float,
                            activation: MaybeActivationBuilder,
            ) -> typing.List[FeedForwardBlockBuilder]:
        block_builder = FeedForwardBlockBuilder(
            dropout = dropout,
            activation = activation,
            hidden_size = hidden_size,
        )

        return [block_builder] * block_count
    
@dataclasses.dataclass
class FeedForwardLayersBuilder(BaseBuilder[BaseFeedForward]):
    hidden_size: int
    block_count: int = 1
    dropout: float = 0.3
    activation: MaybeActivationBuilder = DEFAULT_ACTIVATION

    def build(self) -> BaseFeedForward:
        return FeedForwardLayers(
            dropout = self.dropout,
            activation = self.activation,
            hidden_size = self.hidden_size,
            block_count = self.block_count,
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

        network = FeedForwardLayers(
            hidden_size = kwargs["hidden_size"],
            block_count = kwargs["block_count"],
        )

        output = network(input)

        self.assertEqual(output.size(), input.size())
        
    def test_determined_sizes(self):
        sizes: typing.List[typing.Mapping[str, int]] = [
            {"batch_size": 1, "seq_len": 1, "hidden_size": 1, "block_count": 1},
            {"batch_size": 10, "seq_len": 1, "hidden_size": 1, "block_count": 1},
            {"batch_size": 1, "seq_len": 10, "hidden_size": 1, "block_count": 1},
            {"batch_size": 1, "seq_len": 1, "hidden_size": 10, "block_count": 1},
            {"batch_size": 1, "seq_len": 1, "hidden_size": 1, "block_count": 10},
            {"batch_size": 9, "seq_len": 10, "hidden_size": 11, "block_count": 12},
            {"batch_size": 99, "seq_len": 111, "hidden_size": 127, "block_count": 23},
        ]

        for s in sizes:
            self.assert_sizes(**s)

if __name__ == "__main__":
    logging.getLogger("").setLevel(logging.DEBUG)
    unittest.main()
