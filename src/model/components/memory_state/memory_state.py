import typing

from dataclasses import dataclass

from src.model.utils.typing import FPTensor

@dataclass
class MemoryState:
    memory: FPTensor
    normalization: FPTensor

    def __init__(self, memory: FPTensor, normalization: FPTensor) -> None:
        MemoryState.__check_dimensions(memory, normalization)

        self.memory, self.normalization = memory, normalization

    @staticmethod
    def __check_dimensions(memory, normalization) -> None:
        memory_batch_size, memory_key_dim, _ = memory.size()
        normalization_batch_size, normalization_key_dim = normalization.size()

        assert memory_batch_size == normalization_batch_size
        assert memory_key_dim == normalization_key_dim

    def compare_dimensions(self, other: "MemoryState") -> bool:
        result = self.normalization.size() == other.normalization.size()
        return result and (self.memory.size() == other.memory.size())

    @property
    def key_dim(self) -> int:
        _, memory_key_dim, _ = self.memory.size()
        return memory_key_dim

    @property
    def value_dim(self) -> int:
        _, _, memory_value_dim = self.memory.size()
        return memory_value_dim

    @property
    def batch_size(self) -> int:
        memory_batch_size, _, _ = self.memory.size()
        return memory_batch_size
    
    def size(self, *args, **kwargs) -> typing.Any:
        return self.memory.size(*args, **kwargs)
    
    def __iter__(self) -> FPTensor:
        yield self.memory
        yield self.normalization
