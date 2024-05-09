import abc

import torch

import dataclasses

from src.model.utils.typing import FPTensor
from src.model.utils.builders import BaseBuilder, MaybeBuilder

from src.model.components.activation.base_activation import BaseActivation

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
    
@dataclasses.dataclass
class ELUBuilder(BaseBuilder[ELU]):
    bias: float = 0.0
    alpha: float = 1.0

    def build(self) -> ELU:
        return ELU(
            bias = self.bias,
            alpha = self.alpha,
        )
    
MaybeELUBuilder = MaybeBuilder[ELU]

DEFAULT_ACTIVATION: BaseActivation = ELU(bias = 1.0)
