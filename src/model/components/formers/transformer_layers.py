import typing

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.model.utils.builders import BaseBuilder, MaybeBuilder

from src.model.components.formers.transformer_network import TransformerNetwork

from src.model.components.mha_adapter.mha_adapter import MaybeMHAAdapterBuilder
from src.model.components.feed_forward.base_feed_forward import MaybeFeedForwardBuilder

from src.model.components.masking.attention_masking import (
    AttentionMasking, MaybeAttentionMaskingBuilder, DEFAULT_ATTENTION_MASKING,
)

from src.model.components.former_block.transformer_block import (
    TransformerBlockBuilder, MaybeTransformerBlockBuilder,
)

class TransformerLayers(TransformerNetwork):
    def __init__(self,
                 multiheadattention: MaybeMHAAdapterBuilder,
                 feedforwardnetwork: MaybeFeedForwardBuilder,
                 attention_masking: MaybeAttentionMaskingBuilder = DEFAULT_ATTENTION_MASKING,
                 block_count: int = 1,) -> None:
        blocks = TransformerLayers.make_block_builders(
            multiheadattention = multiheadattention,
            feedforwardnetwork = feedforwardnetwork,
            block_count = block_count,
        )

        super().__init__(
            attention_masking = attention_masking,
            blocks = blocks,
        )


    @staticmethod
    def make_block_builders(
                multiheadattention: MaybeMHAAdapterBuilder,
                feedforwardnetwork: MaybeFeedForwardBuilder,
                block_count: int = 1) -> typing.List[MaybeTransformerBlockBuilder]:
        block_builder = TransformerBlockBuilder(
            multiheadattention = multiheadattention,
            feedforwardnetwork = feedforwardnetwork,
        )

        return [block_builder] * block_count

import torch

import logging
import unittest

from src.model.utils.typing import FPTensor

from src.model.components.mha_adapter.mha_adapter import MHAAdapter
from src.model.components.feed_forward.base_feed_forward import BaseFeedForward
from src.model.components.feed_forward.feed_forward_layers import FeedForwardLayers

log = logging.getLogger(__name__)

class TestTransformerLayers(unittest.TestCase):
    def make_generator(self, **kwargs) -> torch.Generator:
        seed = sum(kwargs.values()) + 777
        generator = torch.Generator()
        generator.manual_seed(seed)
        return generator
    
    def make_batch(self, gen, **kwargs) -> FPTensor:
        size = (kwargs["bs"], kwargs["sl"], kwargs["ed"])
        return torch.rand(size, generator = gen)
    
    def make_multiheadattention(self, **kwargs) -> MHAAdapter:
        return MHAAdapter(
            query_dim = kwargs["ed"],
            head_count = kwargs["hc"],
        )
    
    def make_feedforwardlayers(self, **kwargs) -> BaseFeedForward:
        return FeedForwardLayers(
            hidden_size = kwargs["ed"],
            block_count = kwargs["lc"],
        )
    
    def make_transformerlayers(self, **kwargs) -> TransformerNetwork:
        return TransformerLayers(
            multiheadattention = self.make_multiheadattention(**kwargs),
            feedforwardnetwork = self.make_feedforwardlayers(**kwargs),
            block_count = kwargs["bc"]
        )
    
    def check_sizes(self, **kwargs) -> None:
        gen = self.make_generator(**kwargs)
        batch = self.make_batch(gen, **kwargs)
        block = self.make_transformerlayers(**kwargs)

        result = block(batch)
        
        self.assertEqual(result.size(), batch.size())

    def test_sizes(self) -> None:
        for sizes in self.get_sizes():
            self.check_sizes(**sizes)

    def get_sizes(self) -> typing.List[typing.Dict[str, int]]:
        return [
            {"bs": 1, "sl": 1, "ed": 1, "hc": 1, "lc": 1, "bc": 1},
            {"bs": 2, "sl": 1, "ed": 1, "hc": 1, "lc": 1, "bc": 1},
            {"bs": 1, "sl": 2, "ed": 1, "hc": 1, "lc": 1, "bc": 1},
            {"bs": 1, "sl": 1, "ed": 2, "hc": 1, "lc": 1, "bc": 1},
            {"bs": 1, "sl": 1, "ed": 2, "hc": 2, "lc": 1, "bc": 1},
            {"bs": 1, "sl": 1, "ed": 1, "hc": 1, "lc": 2, "bc": 1},
            {"bs": 1, "sl": 1, "ed": 1, "hc": 1, "lc": 1, "bc": 2},
            {"bs": 2, "sl": 3, "ed": 4, "hc": 2, "lc": 5, "bc": 1},
            {"bs": 3, "sl": 4, "ed": 6, "hc": 2, "lc": 7, "bc": 9},
            {"bs": 4, "sl": 5, "ed": 9, "hc": 3, "lc": 6, "bc": 8},
        ]

if __name__ == "__main__":
    unittest.main()
