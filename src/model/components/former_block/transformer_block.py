import torch
import typing

import dataclasses

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.model.utils.typing import FPTensor
from src.model.utils.builders import value_or_build, BaseBuilder, MaybeBuilder

from src.model.components.mha_adapter.mha_adapter import MHAAdapter, MaybeMHAAdapterBuilder
from src.model.components.feed_forward.base_feed_forward import BaseFeedForward, MaybeFeedForwardBuilder

class TransformerBlock(torch.nn.Module):
    MaybeAttentionMask = typing.Optional[FPTensor]
    MaybePaddingMask = typing.Optional[torch.BoolTensor]

    def __init__(self, 
                 multiheadattention: MaybeMHAAdapterBuilder, 
                 feedforwardnetwork: MaybeFeedForwardBuilder) -> None:
        super().__init__()

        self.__multiheadattention = value_or_build(multiheadattention)
        self.__feedforwardnetwork = value_or_build(feedforwardnetwork)

        input_size = self.feedforwardnetwork.input_size
        self.__layernorm = torch.nn.LayerNorm(input_size)

        assert input_size == self.multiheadattention.key_dim
        assert input_size == self.multiheadattention.query_dim
        assert input_size == self.multiheadattention.value_dim
        
    @property
    def multiheadattention(self) -> MHAAdapter:
        return self.__multiheadattention
    
    @property
    def feedforwardnetwork(self) -> BaseFeedForward:
        return self.__feedforwardnetwork

    @property
    def layernorm(self) -> torch.nn.LayerNorm:
        return self.__layernorm
    
    def apply_layernorm(self, input: FPTensor) -> FPTensor:
        input_size = input.size()
        assert len(input_size) == 3

        output = self.layernorm(input)

        output_size = output.size()
        assert output_size == input_size
        return output
    
    def __validate_padding(self, size, padding) -> None:
        if padding is not None:
            assert padding.size() == size[:-1]

    def __validate_mask(self, size, mask) -> None:
        if mask is not None:
            bs, sl, _ = size
            if mask.dim() == 2:
                assert mask.size() == (sl, sl)
            elif mask.dim() == 2:
                assert mask.size() == (bs, sl, sl)
            else:
                raise ValueError("Invalid mask dimensionality")

    def forward(self, input: FPTensor, 
                attn_mask: MaybeAttentionMask = None,
                padd_mask: MaybePaddingMask = None,) -> FPTensor:
        input_size = input.size()
        assert len(input_size) == 3

        self.__validate_mask(input_size, attn_mask)
        self.__validate_padding(input_size, padd_mask)

        normed = self.apply_layernorm(input)

        attention, _ = self.multiheadattention(
            key_padding_mask = padd_mask,
            queries = normed, values = normed,
            keys = normed, attn_mask = attn_mask,
        )

        assert attention.size() == input_size
        output = self.feedforwardnetwork(input + attention)

        output_size = output.size()
        assert output_size == input_size
        return output
    
@dataclasses.dataclass
class TransformerBlockBuilder(BaseBuilder[TransformerBlock]):
    multiheadattention: MaybeMHAAdapterBuilder
    feedforwardnetwork: MaybeFeedForwardBuilder

    def build(self) -> TransformerBlock:
        return TransformerBlock(
            multiheadattention = self.multiheadattention,
            feedforwardnetwork = self.feedforwardnetwork,
        )
    
MaybeTransformerBlockBuilder = MaybeBuilder[TransformerBlock]

import logging
import unittest

from src.model.components.feed_forward.feed_forward_layers import FeedForwardLayers

log = logging.getLogger(__name__)

class TestTransformerBlock(unittest.TestCase):
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
    
    def make_transformerblock(self, **kwargs) -> TransformerBlock:
        return TransformerBlock(
            multiheadattention = self.make_multiheadattention(**kwargs),
            feedforwardnetwork = self.make_feedforwardlayers(**kwargs),
        )
    
    def check_sizes(self, **kwargs) -> None:
        gen = self.make_generator(**kwargs)
        batch = self.make_batch(gen, **kwargs)
        block = self.make_transformerblock(**kwargs)
        result = block(batch)
        
        self.assertEqual(result.size(), batch.size())

    def test_sizes(self) -> None:
        for sizes in self.get_sizes():
            self.check_sizes(**sizes)

    def get_sizes(self) -> typing.List[typing.Dict[str, int]]:
        return [
            {"bs": 1, "sl": 1, "ed": 1, "hc": 1, "lc": 1},
            {"bs": 2, "sl": 1, "ed": 1, "hc": 1, "lc": 1},
            {"bs": 1, "sl": 2, "ed": 1, "hc": 1, "lc": 1},
            {"bs": 1, "sl": 1, "ed": 2, "hc": 1, "lc": 1},
            {"bs": 1, "sl": 1, "ed": 2, "hc": 2, "lc": 1},
            {"bs": 1, "sl": 1, "ed": 1, "hc": 1, "lc": 2},
            {"bs": 2, "sl": 3, "ed": 4, "hc": 2, "lc": 5},
            {"bs": 3, "sl": 4, "ed": 6, "hc": 2, "lc": 7},
            {"bs": 4, "sl": 5, "ed": 9, "hc": 3, "lc": 6},
        ]

if __name__ == "__main__":
    unittest.main()
