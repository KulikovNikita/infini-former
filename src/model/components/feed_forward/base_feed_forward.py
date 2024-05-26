import abc

import torch

from src.model.utils.typing import FPTensor
from src.model.utils.builders import MaybeBuilder

class BaseFeedForward(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    @property
    @abc.abstractmethod
    def input_size(self) -> int:
        pass
    
    @abc.abstractmethod
    def _forward(self, input: FPTensor) -> FPTensor:
        pass

    def forward(self, input: FPTensor) -> FPTensor:
        input_size = input.size()
        output = self._forward(input)
        assert output.size() == input_size
        return (output + input)

MaybeFeedForwardBuilder = MaybeBuilder[BaseFeedForward]
