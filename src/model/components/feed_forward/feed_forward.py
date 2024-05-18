import torch

import typing
import dataclasses

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.model.utils.typing import FPTensor
from src.model.utils.builders import value_or_build, BaseBuilder

from src.model.components.activation.base_activation import MaybeActivationBuilder
from src.model.components.activation.activations import (
    BaseActivation, DEFAULT_ACTIVATION
)
from src.model.components.feed_forward.feed_forward_block import (
    MaybeFFBlockBuilder, FeedForwardBlock, FeedForwardBlockBuilder
)
from src.model.components.feed_forward.base_feed_forward import BaseFeedForward, BaseFeedForwardBuilder

class FeedForwardNetwork(BaseFeedForward):
    ModuleList = torch.nn.ModuleList
    RawList = typing.List[FeedForwardBlock]
    MaybeBuilderList = typing.List[MaybeFFBlockBuilder]

    def __init__(self, blocks: MaybeBuilderList = []) -> None:
                 
        super().__init__()

        self.__blocks = FeedForwardNetwork.__produce_module(blocks)

    @property
    def blocks(self) -> ModuleList:
        return self.__blocks
    
    def __len__(self) -> int:
        return len(self.blocks)
    
    @property
    def block_count(self) -> int:
        return len(self.blocks)

    @staticmethod
    def __produce_blocks(maybe_blocks: MaybeBuilderList) -> RawList:
        return list(map(value_or_build, maybe_blocks))
    
    @staticmethod
    def __produce_module(maybe_blocks: MaybeBuilderList) -> torch.nn.ModuleList:
        blocks = FeedForwardNetwork.__produce_blocks(maybe_blocks)
        return torch.nn.ModuleList(blocks)

    def _forward(self, sequences: FPTensor) -> FPTensor:
        forwarded = sequences.clone()
        for block in self.blocks[:-1]:
            forwarded = block(forwarded)
        last_block = self.blocks[-1]
        return last_block.forward_wo_activation(forwarded)

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
class FeedForwardLayersBuilder(BaseBuilder[FeedForwardLayers]):
    hidden_size: int
    block_count: int = 1
    dropout: float = 0.3
    activation: MaybeActivationBuilder = DEFAULT_ACTIVATION

    def build(self) -> FeedForwardLayers:
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

        network = FeedForwardLayersBuilder(
            hidden_size = kwargs["hidden_size"],
            block_count = kwargs["block_count"],
        ).build()

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
