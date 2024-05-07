import torch
import typing

from src.model.utils.typing import value_or_default

from src.model.components.feed_forward.base_feed_forward import BaseFeedForward

T = typing.TypeVar('T')

class TransformerBlock(torch.nn.Module):
    def __init__(self,
                 query_dim: int,
                 head_count: int = 2,
                 key_dim: typing.Optional[int] = None,
                 value_dim: typing.Optional[int] = None,
                 feed_forward: BaseFeedForward = ,
                 batch_first: bool = True) -> None:
    
        super().__init__()
    
        self.__query_dim = query_dim
        self.__head_count = head_count
        self.__batch_first = batch_first
        self.__feed_forward = feed_forward

        self.__key_dim = value_or_default(key_dim, self.query_dim)
        self.__value_dim = value_or_default(value_dim, self.value_dim)
        
        self.__attention = self.__make_attention()
    
    def __make_attention(self) -> torch.nn.MultiHeadAtention:
        return torch.nn.MultiHeadAttention(
            kdim = self.key_dim,
            vdim = self.value_dim,
            embed_dim = self.query_dim,
            num_heads = self.head_count,
            batch_first = self.batch_first,
        )
    
    @property
    def __batch_dim(self) -> int:
        return 0 if self.batch_first else 1
    
    @property
    def __sequence_dim(self) -> int:
        return 1 if self.batch_first else 0
    
    @property
    def key_dim(self) -> int:
        return self.__key_dim
    
    @property
    def value_dim(self) -> int:
        return self.__value_dim
    
    @property
    def query_dim(self) -> int:
        return self.__query_dim
