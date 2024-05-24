import torch

import typing

from src.model.utils.typing import FPTensor
from src.model.utils.builders import value_or_build, MaybeBuilder

from src.model.components.adapter.base_adapter import BaseAdapter, MaybeAdapterBuilder

class BaseAttention(torch.nn.Module):
    def __init__(self, 
                 query_adapter: MaybeAdapterBuilder,
                 key_adapter: MaybeAdapterBuilder,
                 value_adapter: MaybeAdapterBuilder) -> None:
        super().__init__()

        self.__query_adapter = value_or_build(query_adapter)
        self.__key_adapter = value_or_build(key_adapter)
        self.__value_adapter = value_or_build(value_adapter)

        self.__common_dim = self.__validate_common_dim()

    def __validate_common_dim(self) -> int:
        query_common = self.query_adapter.common_dim
        key_common = self.key_adapter.common_dim
        assert query_common == key_common
        return key_common

    @property
    def query_dim(self) -> int:
        return self.query_adapter.question_dim
    
    @property
    def query_adapter(self) -> BaseAdapter:
        return self.__query_adapter
    
    @property
    def key_dim(self) -> int:
        return self.key_adapter.question_dim
    
    @property
    def key_adapter(self) -> BaseAdapter:
        return self.__key_adapter
    
    @property
    def value_dim(self) -> int:
        return self.value_adapter.question_dim
    
    @property
    def value_adapter(self) -> BaseAdapter:
        return self.__value_adapter
    
    @property
    def common_dim(self) -> int:
        return self.__common_dim
    
    def _validate_input(self, queries: FPTensor, keys: FPTensor, values: FPTensor) -> None:
        q_batch_size, q_seq_len, q_emb_dim = queries.size()
        v_batch_size, v_seq_len, v_emb_dim = values.size()
        k_batch_size, k_seq_len, k_emb_dim = keys.size()

        assert q_batch_size == v_batch_size
        assert q_batch_size == k_batch_size

        assert q_seq_len == v_seq_len
        assert q_seq_len == k_seq_len

        assert q_emb_dim == self.query_dim
        assert v_emb_dim == self.value_dim
        assert k_emb_dim == self.key_dim

    def _validate_output(self, values: FPTensor, output: FPTensor) -> None:
        o_batch_size, o_seq_len, o_emb_dim = output.size()
        v_batch_size, v_seq_len, _ = values.size()

        assert v_batch_size == o_batch_size
        assert v_seq_len == o_seq_len

        assert o_emb_dim == self.value_dim

    def _prepare_queries(self, queries: FPTensor) -> FPTensor:
        q_size = queries.size()

        result = self.query_adapter(queries)

        r_size = result.size()

        assert r_size[-1] == self.common_dim
        assert q_size[-1] == self.query_dim
        assert r_size[:-1] == q_size[:-1]

        return result
    
    def _prepare_keys(self, keys: FPTensor) -> FPTensor:
        k_size = keys.size()

        result = self.key_adapter(keys)

        r_size = result.size()

        assert r_size[-1] == self.common_dim
        assert k_size[-1] == self.key_dim
        assert r_size[:-1] == k_size[:-1]

        return result
    
    def _prepare_values(self, values: FPTensor) -> FPTensor:
        v_size = values.size()

        result = self.value_adapter(values)

        r_size = result.size()

        assert r_size[-1] == self.value_adapter.common_dim
        assert v_size[-1] == self.value_dim
        assert r_size[:-1] == v_size[:-1]

        return result

    def _prepare_input(self, 
                       queries: FPTensor, 
                       keys: FPTensor, 
                       values: FPTensor,
            ) -> typing.Tuple[FPTensor, FPTensor, FPTensor]:
        self._validate_input(queries, keys, values)

        p_queries = self._prepare_queries(queries)
        p_keys = self._prepare_keys(keys)
        p_values = self._prepare_values(values)

        return (p_queries, p_keys, p_values)
    
MaybeAttentionBuilder = MaybeBuilder[BaseAttention]
