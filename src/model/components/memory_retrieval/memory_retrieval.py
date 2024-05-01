import torch

import typing

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.model.utils.typing import FPTensor
from src.model.utils.activations import BaseActivation, ELU

from src.model.components.memory_state.memory_state import MemoryState

DEFAULT_ACTIVATION: BaseActivation = ELU(bias = 1.0)

class MemoryRetrieval(torch.nn.Module):
    def __init__(self, activation: BaseActivation = DEFAULT_ACTIVATION) -> None:
        super().__init__()

        self.__activation = activation

    @property
    def activation(self) -> BaseActivation:
        return self.__activation
    
    @staticmethod
    def __check_input(state, queries) -> None:
        state_batch_size, state_key_dim, _ = state.size()
        queries_batch_size, _, queries_key_dim = queries.size()

        assert state_batch_size == queries_batch_size
        assert state_key_dim == queries_key_dim
    
    @staticmethod
    def __check_output(state, queries, output) -> None:
        state_batch_size, _, state_value_dim = state.size()
        queries_batch_size, queries_seq_len, _ = queries.size()
        output_batch_size, output_seq_len, output_value_dim = output.size()

        assert output_seq_len == queries_seq_len
        assert output_value_dim == state_value_dim
        assert output_batch_size == state_batch_size
        assert output_batch_size == queries_batch_size

    def __activate_queries(self, queries) -> FPTensor:
        activated_queries = self.activation(queries)
        assert activated_queries.size() == queries.size()
        return activated_queries
    
    def __normalize(self, state, activated_queries) -> FPTensor:
        queries_batch_size, queries_seq_len, _ = activated_queries.size()

        shaped_normalization = state.normalization[:, :, None]
        normalization = torch.matmul(activated_queries, shaped_normalization)
        fixed_normalization = normalization.squeeze(-1)

        assert fixed_normalization.size() == (queries_batch_size, queries_seq_len)
        return fixed_normalization
    
    def __retrieve(self, state, activated_queries) -> FPTensor:
        _, queries_seq_len, _ = activated_queries.size()
        state_batch_size, _, state_value_dim = state.size()

        retrieved = torch.matmul(activated_queries, state.memory)

        assert retrieved.size() == (state_batch_size, queries_seq_len, state_value_dim)

        return retrieved
    
    def __find_delta(self, normalization) -> FPTensor:
        dtype = normalization.dtype
        delta = torch.finfo(dtype).tiny
        delta = torch.Tensor((delta,))
        delta = delta.to(dtype)

        assert delta.size() == (1,)
        assert delta.dtype == dtype

        return delta
    
    def __invert_safely(self, normalization) -> FPTensor:
        delta = self.__find_delta(normalization)
        fixed_normalization = torch.maximum(normalization, delta)
        shaped_normalization = fixed_normalization
        inverted = 1.0 / shaped_normalization

        assert inverted.size() == normalization.size()

        return inverted

    def __apply_normalization(self, retrieved, normalization) -> FPTensor:
        inverted_normalization = self.__invert_safely(normalization)
        applied = retrieved / inverted_normalization[:, :, None]

        return applied

    def forward(self, 
                state: MemoryState, 
                queries: FPTensor) -> FPTensor:
        MemoryRetrieval.__check_input(state, queries)

        activated_queries = self.__activate_queries(queries)

        normalization = self.__normalize(state, activated_queries)

        retrieved = self.__retrieve(state, activated_queries)

        output = self.__apply_normalization(retrieved, normalization)

        MemoryRetrieval.__check_output(state, queries, output)
        return output

import unittest

class TestMemoryRetrieval(unittest.TestCase):
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
    
    def _dummy_queries(self, gen, **kwargs):
        size = (kwargs["batch_size"], kwargs["seq_len"], kwargs["key_dim"])
        return torch.rand(*size, generator = gen, dtype = torch.float32)

    def assert_sizes(self, **kwargs):
        seed = 777 + sum(kwargs.values())
        gen = self._make_generator(seed = seed)
        state = self._dummy_state(gen, **kwargs)
        queries = self._dummy_queries(gen, **kwargs)

        retrieval = MemoryRetrieval()
        result = retrieval(state, queries)

        size = (kwargs["batch_size"], kwargs["seq_len"], kwargs["value_dim"])
        self.assertEqual(result.size(), size)
        
    def test_determined_sizes(self):
        sizes: typing.List[typing.Mapping[str, int]] = [
            {"batch_size": 1, "seq_len": 1, "key_dim": 1, "value_dim": 1},
            {"batch_size": 10, "seq_len": 1, "key_dim": 1, "value_dim": 1},
            {"batch_size": 1, "seq_len": 10, "key_dim": 1, "value_dim": 1},
            {"batch_size": 1, "seq_len": 1, "key_dim": 10, "value_dim": 1},
            {"batch_size": 1, "seq_len": 1, "key_dim": 1, "value_dim": 10},
            {"batch_size": 9, "seq_len": 10, "key_dim": 11, "value_dim": 12},
        ]

        for s in sizes:
            self.assert_sizes(**s)

if __name__ == "__main__":
    unittest.main()
