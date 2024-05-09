import abc

import torch

from src.model.utils.typing import FPTensor
from src.model.utils.builders import value_or_build

from src.model.components.activation.base_activation import (
    BaseActivation, MaybeActivationBuilder,
)

from src.model.components.memory_state.memory_state import MemoryState
from src.model.components.memory_retrieval.memory_retrieval import MemoryRetrieval

class BaseMemoryUpdate(torch.nn.Module):
    def __init__(self, activation: MaybeActivationBuilder) -> None:
        super().__init__()
        self.__activation = value_or_build(activation)

    @property
    def activation(self) -> BaseActivation:
        return self.__activation

    @abc.abstractmethod
    def _forward(self, state: MemoryState, keys: FPTensor, values: FPTensor) -> MemoryState:
        pass

    @staticmethod
    def __check_input(state, keys, values) -> None:
        assert True

    @staticmethod
    def __check_output(state, keys, values, output) -> None:
        assert state.compare_dimensions(output)

    def forward(self, state: MemoryState, keys: FPTensor, values: FPTensor) -> MemoryState:
        BaseMemoryUpdate.__check_input(state, keys, values)

        output: MemoryState = self._forward(state, keys, values)

        BaseMemoryUpdate.__check_output(state, keys, values, output)
        return output
