import torch

import typing
import dataclasses

from src.model.utils.typing import FPTensor
from src.model.utils.builders import value_or_build, BaseBuilder

from src.model.components.feed_forward.feed_forward_block import (
    MaybeFFBlockBuilder, FeedForwardBlock,
)
from src.model.components.feed_forward.base_feed_forward import BaseFeedForward

class FeedForwardNetwork(BaseFeedForward):
    ModuleList = torch.nn.ModuleList
    RawList = typing.List[FeedForwardBlock]
    MaybeBuilderList = typing.List[MaybeFFBlockBuilder]

    def __init__(self, blocks: MaybeBuilderList) -> None:
        super().__init__()
        self.__blocks = FeedForwardNetwork.__produce_module(blocks)
        self.__input_size = self.blocks[0].hidden_size

    @property
    def blocks(self) -> ModuleList:
        return self.__blocks
    
    def __len__(self) -> int:
        return len(self.blocks)
    
    def input_size(self) -> int:
        return self.__input_size
    
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
    
@dataclasses.dataclass
class FeedForwardNetworkBuilder(BaseBuilder[FeedForwardNetwork]):
    blocks: typing.List[MaybeFFBlockBuilder]

    def build(self) -> BaseFeedForward:
        return FeedForwardNetwork(
            blocks = self.blocks,
        )
