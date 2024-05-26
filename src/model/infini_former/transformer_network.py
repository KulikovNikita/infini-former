import torch
import typing

import dataclasses

from src.model.utils.typing import FPTensor
from src.model.utils.builders import value_or_build, BaseBuilder, MaybeBuilder

from src.model.components.masking import AttentionMasking
from src.model.components.former_block.transformer_block import (
    TransformerBlock, MaybeTransformerBlockBuilder,
)

class TransformerNetwork(torch.nn.Module):
    BlockList = typing.List[TransformerBlock]
    MaybeBlockList = typing.List[MaybeTransformerBlockBuilder]

    def __init__(self, blocks: MaybeBlockList) -> None:
        super().__init__()

        self.__masking = AttentionMasking()
        self.__blocks = TransformerNetwork.__prepare_module(blocks)

    @staticmethod
    def __prepare_blocks(blocks: MaybeBlockList) -> BlockList:
        return list(map(value_or_build, blocks))
    
    @staticmethod
    def __prepare_module(blocks: MaybeBlockList) -> torch.nn.Module:
        blocks = TransformerNetwork.__prepare_blocks(blocks)
        return torch.nn.ModuleList(blocks)
    
    @property
    def blocks(self) -> torch.nn.Module:
        return self.__blocks
    
    @property
    def masking(self) -> AttentionMasking:
        return self.__masking

    def forward(self, input: FPTensor) -> FPTensor:
        input_size = input.size()
        assert len(input_size) == 3
        mask = self.masking(input_size)

        output = input.clone()
        for block in self.blocks:
            output = block(output, mask)
            assert output.size() == input_size

        return output

@dataclasses.dataclass
class TransformerNetworkBuilder(BaseBuilder[TransformerNetwork]):
    blocks: typing.List[MaybeTransformerBlockBuilder]

    def build(self) -> TransformerNetwork:
        return TransformerNetwork(
            blocks = self.blocks
        )
    
MaybeTransformerNetworkBuilder = MaybeBuilder[TransformerNetwork]
