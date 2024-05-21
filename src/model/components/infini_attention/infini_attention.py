import torch

import typing

from src.model.utils.typing import value_or_default, FPTensor

class InfiniAttention(torch.nn.Module):
    MaybeAdapter = typing.Optional[torch.nn.Linear]

    def __init__(self, 
                 query_dim: int,
                 key_dim: typing.Optional[int] = None,
                 value_dim: typing.Optional[int] = None,
                 common_dim: typing.Optional[int] = None) -> None:
        
        super().__init__()

        self.__query_dim = query_dim
        self.__key_dim = value_or_default(key_dim, query_dim)
        self.__value_dim = value_or_default(value_dim, query_dim)

        default_common = max(key_dim, query_dim)
        self.__common_dim = value_or_default(common_dim, default_common)
    
        self.__key_adapter = self.__make_key_adapter()
        self.__query_adapter = self.__make_query_adapter()

    @property
    def key_dim(self) -> int:
        return self.__key_dim
    
    @property
    def value_dim(self) -> int:
        return self.__value_dim
    
    @property
    def query_dim(self) -> int:
        return self.__query_dim
    
    @property
    def common_dim(self) -> int:
        return self.__common_dim
    
    @property
    def key_adapter(self) -> MaybeAdapter: 
        return self.__key_adapter
    
    @property
    def query_adapter(self) -> MaybeAdapter:
        return self.__query_adapter
    
    def adapt_keys(self, keys: FPTensor) -> FPTensor:
        if self.key_adapter is not None:
            return self.key_adapter(keys)
        else:
            return keys

    def adapt_queries(self, queries: FPTensor) -> FPTensor:
        if self.query_adapter is not None:
            result = self.query_adapter(queries)
        else:
            result = queries

        return result
        
    def forward(self, keys, queries, values, attention) -> FPTensor:
 

class InfiniMHA(torch.nn.Module):
    pass
