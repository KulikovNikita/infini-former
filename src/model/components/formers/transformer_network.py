import torch
import typing

import dataclasses

from src.model.utils.typing import FPTensor
from src.model.utils.builders import value_or_build, BaseBuilder, MaybeBuilder

from src.model.components.masking.attention_masking import (
    AttentionMasking, MaybeAttentionMaskingBuilder, DEFAULT_ATTENTION_MASKING,
)
from src.model.components.former_block.transformer_block import (
    TransformerBlock, MaybeTransformerBlockBuilder,
)

class TransformerNetwork(torch.nn.Module):
    BlockList = typing.List[TransformerBlock]
    OptionalPadding = typing.Optional[torch.BoolTensor]
    MaybeBlockList = typing.List[MaybeTransformerBlockBuilder]

    def __init__(self, 
                 blocks: MaybeBlockList,
                 attention_masking: MaybeAttentionMaskingBuilder = DEFAULT_ATTENTION_MASKING) -> None:
        super().__init__()

        self.__attention_masking = value_or_build(attention_masking)
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
    def attention_masking(self) -> AttentionMasking:
        return self.__attention_masking

    def forward(self, input: FPTensor, padding: OptionalPadding = None) -> FPTensor:
        input_size = input.size()
        assert len(input_size) == 3
        mask = self.attention_masking(input_size)

        output = input.clone()
        for block in self.blocks:
            output = block(output, mask, padding)
            assert output.size() == input_size

        return output

@dataclasses.dataclass
class TransformerNetworkBuilder(BaseBuilder[TransformerNetwork]):
    blocks: typing.List[MaybeTransformerBlockBuilder]
    attention_masking: MaybeAttentionMaskingBuilder = DEFAULT_ATTENTION_MASKING

    def build(self) -> TransformerNetwork:
        return TransformerNetwork(
            blocks = self.blocks,
            attention_masking = self.attention_masking,
        )
    
MaybeTransformerNetworkBuilder = MaybeBuilder[TransformerNetwork]
