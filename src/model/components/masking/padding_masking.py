import abc

import torch

import typing

import dataclasses

from src.model.utils.typing import IndexTensor
from src.model.utils.builders import BaseBuilder, MaybeBuilder

class BasePaddingMasking(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    @abc.abstractmethod
    def _forward(self, tokens: IndexTensor) -> torch.BoolTensor:
        pass

    def forward(self, tokens: IndexTensor) -> torch.BoolTensor:
        batch_size, seq_len = tokens.size()

        result = self._forward(tokens)

        assert result.size() == (batch_size, seq_len)
        return result

MaybePaddingMaskingBuilder = MaybeBuilder[BasePaddingMasking]

class SetPaddingMasking(BasePaddingMasking):
    ElementsInput = typing.Sequence[int]

    def __init__(self, mask_elements: ElementsInput = {}) -> None:
        super().__init__()

        self.__mask_elements = set(mask_elements)

    @property
    def mask_elements(self) -> typing.Set[int]:
        return self.__mask_elements
    
    def _forward(self, tokens: IndexTensor) -> torch.BoolTensor:
        masks = torch.ones_like(tokens, dtype = torch.bool)

        for element in self.mask_elements:
            masks &= (tokens != element)

        return masks
    
@dataclasses.dataclass
class SetPaddingMaskingBuilder(BaseBuilder[BasePaddingMasking]):
    mask_elements: typing.Set[int] = dataclasses.field(
        default_factory=set,
    )

    def build(self) -> BasePaddingMasking:
        return SetPaddingMasking(
            mask_elements = self.mask_elements,
        )
