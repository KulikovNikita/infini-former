import torch

import typing

import dataclasses

from src.model.utils.typing import value_or_default, FPTensor
from src.model.utils.builders import BaseBuilder, MaybeBuilder

class MHAAdapter(torch.nn.Module):
    def __init__(self, 
                 query_dim: int,
                 head_count: int = 2,
                 bias: bool = True,
                 dropout: float = 0.0,
                 key_dim: typing.Optional[int] = None,
                 value_dim: typing.Optional[int] = None,
                 add_zero_attention: bool = False,
                 add_bias_key_value: bool = False,) -> None:
        super().__init__()

        self.__bias = bias
        self.__dropout = dropout
        self.__head_count = head_count
        self.__add_zero_attention = add_zero_attention
        self.__add_bias_key_value = add_bias_key_value

        self.__query_dim = query_dim
        self.__key_dim = value_or_default(key_dim, query_dim)
        self.__value_dim = value_or_default(value_dim, query_dim)


        self.__implementation = self.__make_implementation()

    @property
    def query_dim(self) -> int:
        return self.__query_dim
    
    @property
    def head_count(self) -> int:
        return self.__head_count
    
    @property
    def bias(self) -> bool:
        return self.__bias
    
    @property
    def dropout(self) -> float:
        return self.__dropout
    
    @property
    def key_dim(self) -> int:
        return self.__key_dim
    
    @property
    def value_dim(self) -> int:
        return self.__value_dim
    
    @property
    def add_zero_attention(self) -> bool:
        return self.__add_zero_attention
    
    @property
    def add_bias_key_value(self) -> bool:
        return self.__add_bias_key_value

    def __make_implementation(self) -> torch.nn.MultiheadAttention:
        return torch.nn.MultiheadAttention(
            embed_dim = self.query_dim,
            num_heads = self.head_count,
            bias = self.bias,
            dropout = self.dropout,
            kdim = self.key_dim,
            vdim = self.value_dim,
            add_zero_attn = self.add_zero_attention,
            add_bias_kv = self.add_bias_key_value,
            batch_first = True, # Internal agreement
        )

    @property
    def implementation(self) -> torch.nn.MultiheadAttention:
        return self.__implementation
    
    def __validate_input(self, queries, keys, values) -> None:
        pass

    def __validate_output(self, keys, result) -> None:
        pass
    
    def forward(self, queries: FPTensor, keys: FPTensor, values: FPTensor, **kwargs) -> typing.Tuple[FPTensor, FPTensor]:
        self.__validate_input(queries, keys, values)

        result = self.implementation(queries, keys, values, **kwargs)

        self.__validate_output(keys, result)

        return result

@dataclasses.dataclass
class MHAAdapterBuilder(BaseBuilder[MHAAdapter]):
    query_dim: int
    head_count: int = 2
    bias: bool = True
    dropout: float = 0.0
    key_dim: typing.Optional[int] = None
    value_dim: typing.Optional[int] = None
    add_zero_attention: bool = False
    add_bias_key_value: bool = False

    def build(self) -> MHAAdapter:
        return MHAAdapter(
            query_dim = self.query_dim,
            head_count = self.head_count,
            bias = self.bias,
            dropout = self.dropout,
            key_dim = self.key_dim,
            value_dim = self.value_dim,
            add_zero_attention = self.add_zero_attention,
            add_bias_key_value = self.add_bias_key_value,
        )
    
MaybeMHAAdapterBuilder = MaybeBuilder[MHAAdapter]
