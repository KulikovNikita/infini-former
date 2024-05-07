import abc

import torch

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.model.utils.typing import FPTensor

class BaseFeedForward(torch.nn.Module):
    def __init__(self, layers: torch.nn.ModuleList) -> None:
        super().__init__()
        self.__layers = layers

    @property
    def layer_count(self) -> int:
        return len(self.layers)

    @property
    def layers(self) -> torch.nn.ModuleList:
        return self.__layers
    
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
