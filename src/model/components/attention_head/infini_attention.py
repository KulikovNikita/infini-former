import torch
import typing

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.model.utils.typing import FPTensor
from src.model.utils.builders import BaseBuilder, value_or_build

from src.model.components.memory_state.memory_state import MemoryState
from src.model.components.memory_update.memory_update import BaseMemoryUpdate, MaybeMemoryUpdateBuilder
from src.model.components.memory_update.delta_memory_update import DEFAULT_MEMORY_UPDATE
from src.model.components.memory_retrieval.memory_retrieval import (
    MemoryRetrieval, MaybeMemoryRetrievalBuilder, DEFAULT_MEMORY_RETRIEVAL,
)

from src.model.components.attention_head.base_attention import BaseAttention, MaybeAttentionBuilder

class InfiniAttentionHead(BaseAttention):
    Output = typing.Tuple[FPTensor, MemoryState]
    MaybeMask = typing.Optional[torch.BoolTensor]
    QKV = typing.Tuple[FPTensor, FPTensor, FPTensor]

    def __init__(self, 
                 base_attention: MaybeAttentionBuilder,
                 memory_update: MaybeMemoryUpdateBuilder = DEFAULT_MEMORY_UPDATE,
                 memory_retrieval: MaybeMemoryRetrievalBuilder = DEFAULT_MEMORY_RETRIEVAL,
                 default_mix_value: float = 0.5) -> None:
        built_attention = value_or_build(base_attention)
        super().__init__(
            query_adapter = built_attention.query_adapter,
            value_adapter = built_attention.value_adapter,
            key_adapter = built_attention.key_adapter,
        )
        
        self.__memory_update = value_or_build(memory_update)
        self.__base_attention = value_or_build(built_attention)
        self.__memory_retrieval = value_or_build(memory_retrieval)

        self.register_mix_parameter(default_mix = default_mix_value)

    @property
    def attention(self) -> BaseAttention:
        return self.__attention
    
    @property
    def memory_update(self) -> BaseMemoryUpdate:
        return self.__memory_update
    
    @property
    def memory_retrieval(self) -> MemoryRetrieval:
        return self.__memory_retrieval
    
    @property
    def base_attention(self) -> BaseAttention:
        return self.__base_attention

    def register_mix_parameter(self, default_mix: float) -> None:
        mix_tensor = torch.Tensor((default_mix,))
        self.mix = torch.nn.Parameter(mix_tensor)

    def mix_attentions(self, alpha, beta) -> FPTensor:
        nl_mix: FPTensor = torch.sigmoid(self.mix)
        return alpha * nl_mix + beta * (1.0 - nl_mix)

    def forward(self, qkv: QKV, state: MemoryState, mask: MaybeMask = None) -> Output:
        queries, keys, values = qkv
        self._validate_input(queries, keys, values)

        q, k, v = self._prepare_input(queries, keys, values)

        updated_memory = self.memory_update(state, k, v)
        memory_attention = self.memory_retrieval(state, q)
        base_attention = self.base_attention.forward_prepared(q, k, v, mask)
        mixed_attention = self.mix_attentions(memory_attention, base_attention)

        return (mixed_attention, updated_memory)

import logging
import unittest

from src.model.components.attention_head.attention_head import DefaultAttentionHead

log = logging.getLogger(__name__)

class TestInfiniAttentionHead(unittest.TestCase):
    def test_dimensions(self) -> None:
        for d in self.get_dimensions():
            d["cd"] = max(d["kd"], d["qd"])
            self.check_dimension(**d)

    def make_generator(self, **kwargs) -> torch.Generator:
        seed = sum(kwargs.values()) + 777
        gen = torch.Generator()
        gen.manual_seed(seed)
        return gen
    
    def check_size(self, result, **kwargs) -> None:
        pass
    
    def gen_keys(self, gen, **kwargs) -> FPTensor:
        size = (kwargs["bs"], kwargs["sl"], kwargs["kd"])
        return torch.rand(*size, generator = gen)
    
    def gen_values(self, gen, **kwargs) -> FPTensor:
        size = (kwargs["bs"], kwargs["sl"], kwargs["vd"])
        return torch.rand(*size, generator = gen)
    
    def gen_queries(self, gen, **kwargs) -> FPTensor:
        size = (kwargs["bs"], kwargs["sl"], kwargs["qd"])
        return torch.rand(*size, generator = gen)
    
    def make_attention(self, **kwargs) -> InfiniAttentionHead:
        base = DefaultAttentionHead(
            key_dim = kwargs["kd"], 
            value_dim = kwargs["vd"],
            query_dim = kwargs["qd"],
        )
        return InfiniAttentionHead(
            base_attention = base,
            memory_update = DEFAULT_MEMORY_UPDATE,
            memory_retrieval = DEFAULT_MEMORY_RETRIEVAL, 
        )
    
    def gen_normalization(self, gen, **kwargs):
        size = (kwargs["bs"], kwargs["cd"])
        return torch.rand(*size, generator = gen)
    
    def gen_memory(self, gen, **kwargs):
        size = (kwargs["bs"], kwargs["cd"], kwargs["vd"])
        return torch.rand(*size, generator = gen)
    
    def gen_state(self, gen, **kwargs):
        return MemoryState(
            memory = self.gen_memory(gen, **kwargs),
            normalization = self.gen_normalization(gen, **kwargs), 
        )

    def check_dimension(self, **kwargs) -> None:
        gen = self.make_generator(**kwargs)
        keys = self.gen_keys(gen, **kwargs)
        state = self.gen_state(gen, **kwargs)
        values = self.gen_values(gen, **kwargs)
        queries = self.gen_queries(gen, **kwargs)
        attention = self.make_attention(**kwargs)
        qkv: typing.Tuple = (queries, keys, values)
        result = attention(qkv, state)
        self.check_size(result, **kwargs)

    def get_dimensions(self) -> typing.List[typing.Dict[str, int]]:
        return [
            {"bs": 1, "sl": 1, "kd": 1, "qd": 1, "vd": 1},
            {"bs": 2, "sl": 1, "kd": 1, "qd": 1, "vd": 1},
            {"bs": 1, "sl": 3, "kd": 1, "qd": 1, "vd": 1},
            {"bs": 1, "sl": 1, "kd": 4, "qd": 1, "vd": 1},
            {"bs": 1, "sl": 1, "kd": 1, "qd": 5, "vd": 1},
            {"bs": 1, "sl": 1, "kd": 1, "qd": 1, "vd": 6},
            {"bs": 1, "sl": 2, "kd": 3, "qd": 4, "vd": 5},
            {"bs": 2, "sl": 3, "kd": 4, "qd": 5, "vd": 6},
            {"bs": 3, "sl": 4, "kd": 5, "qd": 6, "vd": 7},
            {"bs": 4, "sl": 5, "kd": 6, "qd": 7, "vd": 8},
            {"bs": 5, "sl": 6, "kd": 7, "qd": 8, "vd": 9},
        ]

if __name__ == "__main__":
    unittest.main()
