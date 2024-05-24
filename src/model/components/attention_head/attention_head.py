import torch
import typing

import dataclasses

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.model.utils.typing import value_or_default, FPTensor
from src.model.utils.builders import MaybeBuilder, BaseBuilder

from src.model.components.adapter.base_adapter import MaybeAdapterBuilder
from src.model.components.attention_head.base_attention import BaseAttention
from src.model.components.adapter.default_adapter import DefaultAdapterBuilder

class AttentionHead(BaseAttention):
    MaybeMask = typing.Optional[torch.BoolTensor]

    def __init__(self, 
                 query_adapter: MaybeAdapterBuilder,
                 key_adapter: MaybeAdapterBuilder,
                 value_adapter: MaybeAdapterBuilder) -> None:
        super().__init__(query_adapter, key_adapter, value_adapter)
        self.__softmax = torch.nn.Softmax(dim = -1)

    @property
    def softmax(self) -> torch.nn.Softmax:
        return self.__softmax

    def __mask_weights(self, weights: FPTensor, mask: MaybeMask) -> FPTensor:
        if mask is None:
            result = weights
        else:
            self._validate_weights(weights, mask)
            mask_3d = mask[:, :, None]
            result = weights + mask_3d

        assert result.size() == weights.size()
        return result
        
    def _validate_weights(self, weights, mask) -> None:
        _, w_q_count, w_k_count = weights.size()
        m_q_count, m_k_count = mask.size()

        assert w_q_count == m_q_count
        assert w_k_count == m_k_count

    def forward_prepared(self, q, k, v, mask: MaybeMask = None) -> FPTensor:
        qk_coefficient = torch.Tensor((1.0 / self.common_dim,))
        qk_coefficient = torch.sqrt(qk_coefficient).to(q.device)

        raw_weights = torch.bmm(q, k.mT) * qk_coefficient

        masked_weights = self.__mask_weights(raw_weights, mask)
        softmax_weights = self.softmax(masked_weights)
        result = torch.bmm(softmax_weights, v)

        self._validate_output(v, result)
        return result

    def forward(self, queries, keys, values, mask: MaybeMask = None) -> FPTensor:
        self._validate_input(queries, keys, values)

        q, k, v = self._prepare_input(queries, keys, values)
        result = self.forward_prepared(q, k, v, mask)

        self._validate_output(v, result)
        return result
    
@dataclasses.dataclass
class AttentionHeadBuilder(BaseBuilder[BaseAttention]):
    query_adapter: MaybeAdapterBuilder
    key_adapter: MaybeAdapterBuilder
    value_adapter: MaybeAdapterBuilder

    def build(self) -> BaseAttention:
        return AttentionHead(
            query_adapter = self.query_adapter,
            key_adapter = self.key_adapter,
            value_adapter = self.value_adapter
        )
    
MaybeAttentionHeadBuilder = MaybeBuilder[AttentionHead]
    
class DefaultAttentionHead(AttentionHead):
    def __init__(self, 
                 query_dim: int, 
                 key_dim: typing.Optional[int] = None, 
                 value_dim: typing.Optional[int] = None) -> None:
        k_dim = value_or_default(key_dim, query_dim)
        v_dim = value_or_default(value_dim, query_dim)

        common_dim = max(query_dim, k_dim)
        query_adapter = DefaultAdapterBuilder(query_dim, common_dim)
        key_adapter = DefaultAdapterBuilder(k_dim, common_dim)
        value_adapter = DefaultAdapterBuilder(v_dim, v_dim)

        super().__init__(query_adapter, key_adapter, value_adapter)

        assert self.common_dim == common_dim
    
import logging
import unittest

log = logging.getLogger(__name__)

class TestAttentionHead(unittest.TestCase):
    def test_dimensions(self) -> None:
        for d in self.get_dimensions():
            self.check_dimension(**d)

    def make_generator(self, **kwargs) -> torch.Generator:
        seed = sum(kwargs.values()) + 777
        gen = torch.Generator()
        gen.manual_seed(seed)
        return gen
    
    def check_size(self, result, **kwargs) -> None:
        r_bs, r_sl, r_vd = result.size()
        assert r_bs == kwargs["bs"]
        #assert r_sl == 
    
    def gen_keys(self, gen, **kwargs) -> FPTensor:
        size = (kwargs["bs"], kwargs["sl"], kwargs["kd"])
        return torch.rand(*size, generator = gen)
    
    def gen_values(self, gen, **kwargs) -> FPTensor:
        size = (kwargs["bs"], kwargs["sl"], kwargs["vd"])
        return torch.rand(*size, generator = gen)
    
    def gen_queries(self, gen, **kwargs) -> FPTensor:
        size = (kwargs["bs"], kwargs["sl"], kwargs["qd"])
        return torch.rand(*size, generator = gen)
    
    def make_attention(self, **kwargs) -> AttentionHead:
        result = DefaultAttentionHead(
            key_dim = kwargs["kd"], 
            value_dim = kwargs["vd"],
            query_dim = kwargs["qd"],
        )
        return result

    def check_dimension(self, **kwargs) -> None:
        gen = self.make_generator(**kwargs)
        keys = self.gen_keys(gen, **kwargs)
        values = self.gen_values(gen, **kwargs)
        queries = self.gen_queries(gen, **kwargs)
        attention = self.make_attention(**kwargs)
        result = attention(queries, keys, values)
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
