import abc

import torch

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.model.utils.typing import FPTensor

class BaseFeedForward(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    @abc.abstractmethod
    def _forward(self, input: FPTensor) -> FPTensor:
        pass

    def forward(self, input: FPTensor) -> FPTensor:
        input_size = input.size()
        output = self._forward(input)
        assert output.size() == input_size
        return (output + input)
    
class BaseFeedForwardBuilder:
    @abc.abstractmethod
    def build(self) -> BaseFeedForward:
        pass
