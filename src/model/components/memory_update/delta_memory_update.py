import torch

import typing

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.model.utils.typing import FPTensor

from src.model.components.activation.activations import DEFAULT_ACTIVATION
from src.model.components.activation.base_activation import MaybeActivationBuilder

from src.model.components.memory_state.memory_state import MemoryState
from src.model.components.memory_update.memory_update import BaseMemoryUpdate
from src.model.components.memory_retrieval.memory_retrieval import MemoryRetrieval

class DeltaMemoryUpdate(BaseMemoryUpdate):
    def __init__(self, activation: MaybeActivationBuilder = DEFAULT_ACTIVATION) -> None:
        super().__init__(activation = activation)
        self.__retrieval = MemoryRetrieval(activation)

    @property
    def retrieval(self) -> MemoryRetrieval:
        return self.__retrieval

    def _update_memory(self, state, activated, values) -> FPTensor:
        state_batch_size, state_key_dim, state_value_dim = state.size()
        retrieved = self.retrieval.retrieve_activated(state, activated)

        assert retrieved.size() == values.size()

        activated_fixed = torch.permute(activated, (0, 2, 1))

        memory_delta = torch.matmul(activated_fixed, retrieved + values)

        output = state.memory + memory_delta

        return output

    def _update_normalization(self, state, activated) -> FPTensor:
        state_batch_size, state_key_dim, _ = state.size()

        activated_fixed = torch.permute(activated, (0, 2, 1))
        normalization_delta = torch.sum(activated_fixed, dim = 2)

        assert normalization_delta.size() == state.normalization.size()
        assert normalization_delta.size() == (state_batch_size, state_key_dim)

        output = state.normalization + normalization_delta

        assert output.size() == normalization_delta.size()

        return output

    def _forward(self, state: MemoryState, keys: FPTensor, values: FPTensor) -> MemoryState:
        activated = self.activation(keys)

        updated_memory = self._update_memory(state, activated, values)
        updated_normalization = self._update_normalization(state, activated)

        output = MemoryState(updated_memory, updated_normalization)

        return output

import unittest

class TestDeltaMemoryUpdate(unittest.TestCase):
    def _make_generator(self, seed = 777):
        gen = torch.Generator()
        return gen.manual_seed(seed)
    
    def _dummy_normalization(self, gen, **kwargs):
        size = (kwargs["batch_size"], kwargs["key_dim"])
        return torch.rand(*size, generator = gen, dtype = torch.float32)
    
    def _dummy_memory(self, gen, **kwargs):
        size = (kwargs["batch_size"], kwargs["key_dim"], kwargs["value_dim"])
        return torch.rand(*size, generator = gen, dtype = torch.float32)
    
    def _dummy_state(self, gen, **kwargs):
        return MemoryState(
            memory = self._dummy_memory(gen, **kwargs),
            normalization = self._dummy_normalization(gen, **kwargs), 
        )
    
    def _dummy_keys(self, gen, **kwargs):
        size = (kwargs["batch_size"], kwargs["seq_len"], kwargs["key_dim"])
        return torch.rand(*size, generator = gen, dtype = torch.float32)
    
    def _dummy_values(self, gen, **kwargs):
        size = (kwargs["batch_size"], kwargs["seq_len"], kwargs["value_dim"])
        return torch.rand(*size, generator = gen, dtype = torch.float32)

    def assert_sizes(self, **kwargs):
        seed = 777 + sum(kwargs.values())
        gen = self._make_generator(seed = seed)
        keys = self._dummy_keys(gen, **kwargs)
        state = self._dummy_state(gen, **kwargs)
        values = self._dummy_values(gen, **kwargs)

        update = DeltaMemoryUpdate()
        result = update(state, keys, values)

        normalization_size = (kwargs["batch_size"], kwargs["key_dim"])
        self.assertEqual(result.normalization.size(), normalization_size)

        memory_size = (kwargs["batch_size"], kwargs["key_dim"], kwargs["value_dim"])
        self.assertEqual(result.memory.size(), memory_size)
        
    def test_determined_sizes(self):
        sizes: typing.List[typing.Mapping[str, int]] = [
            {"batch_size": 1, "seq_len": 1, "key_dim": 1, "value_dim": 1},
            {"batch_size": 10, "seq_len": 1, "key_dim": 1, "value_dim": 1},
            {"batch_size": 1, "seq_len": 10, "key_dim": 1, "value_dim": 1},
            {"batch_size": 1, "seq_len": 1, "key_dim": 10, "value_dim": 1},
            {"batch_size": 1, "seq_len": 1, "key_dim": 1, "value_dim": 10},
            {"batch_size": 9, "seq_len": 10, "key_dim": 11, "value_dim": 12},
            {"batch_size": 99, "seq_len": 111, "key_dim": 127, "value_dim": 234},
        ]

        for s in sizes:
            self.assert_sizes(**s)

if __name__ == "__main__":
    unittest.main()
