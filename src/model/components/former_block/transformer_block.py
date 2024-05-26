import torch
import typing

import dataclasses

from src.model.utils.typing import FPTensor
from src.model.utils.builders import value_or_build, BaseBuilder, MaybeBuilder

from src.model.components.mha_adapter.mha_adapter import MHAAdapter, MaybeMHAAdapterBuilder
from src.model.components.feed_forward.base_feed_forward import BaseFeedForward, MaybeFeedForwardBuilder

class TransformerBlock(torch.nn.Module):
    MaybeMask = typing.Optional[FPTensor]

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

    def forward(self, input: FPTensor, attn_mask: MaybeMask = None) -> FPTensor:
        input_size = input.size()
        assert len(input_size) == 3

        normed = self.apply_layernorm(input)

        attention, _ = self.multiheadattention(
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

log = logging.getLogger(__name__)

class TestTransformerBlock(unittest.TestCase):
    def make_

if __name__ == "__main__":
    unittest.main()
