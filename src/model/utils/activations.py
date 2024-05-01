import abc

import torch

import typing

from src.model.utils.typing import FPTensor

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

class ELU(BaseActivation):
    def __init__(self, alpha: float = 1.0, bias: float = 0.0) -> None:
        super().__init__(bias = bias)

        self.__alpha = alpha
        self.__elu = torch.nn.ELU(alpha)

    @property
    def alpha(self) -> float:
        return self.__alpha

    def _forward(self, input: FPTensor) -> FPTensor:
        return self.__elu(input)
    
ACTIVATION_LOOKUP: typing.Mapping[str, type] = {
    "elu": ELU,
}
    
def make_activation(name: str, *args, **kwargs) -> BaseActivation:
    type_class = ACTIVATION_LOOKUP[name]
    return type_class(*args, **kwargs)

DEFAULT_ACTIVATION: BaseActivation = ELU(bias = 1.0)
