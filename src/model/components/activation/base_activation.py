import abc

import torch

from src.model.utils.typing import FPTensor
from src.model.utils.builders import MaybeBuilder

class BaseActivation(torch.nn.Module):
    def __init__(self, bias: float) -> None:
        super().__init__()

        self.__bias = bias

    def __check_output(input: FPTensor, output: FPTensor) -> None:
        assert input.size() == output.size()
        assert input.dtype == output.dtype

    @property
    def bias(self) -> float:
        return self.__bias

    @abc.abstractmethod
    def _forward(self, input: FPTensor) -> FPTensor:
        pass

    def forward(self, input: FPTensor) -> FPTensor:
        result = self._forward(input) + self.bias
        BaseActivation.__check_output(input, result)
        return result

MaybeActivationBuilder = MaybeBuilder[BaseActivation]
