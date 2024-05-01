import abc

import torch

from src.model.utils.typing import FPTensor
from src.model.utils.activations import BaseActivation, ELU

from src.model.components.memory_state.memory_state import MemoryState
from src.model.components.memory_retrieval.memory_retrieval import MemoryRetrieval

class BaseMemoryUpdate(torch.nn.Module):
    def __init__(self, activation: BaseActivation) -> None:
        super().__init__()
        self.__activation = activation

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
