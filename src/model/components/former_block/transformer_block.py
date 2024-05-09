import torch
import typing

from src.model.utils.typing import value_or_default, FPTensor

from src.model.components.feed_forward.base_feed_forward import BaseFeedForward

class TransformerBlock(torch.nn.Module):
    def __init__(self,
                 query_dim: int,
                 feed_forward: BaseFeedForward,
                 head_count: int = 2,
                 key_dim: typing.Optional[int] = None,
                 value_dim: typing.Optional[int] = None) -> None:
    
        super().__init__()
    
        self.__query_dim = query_dim
        self.__head_count = head_count
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
    
    @property
    def head_count(self) -> int:
        return self.__head_count
    
    @property
    def batch_first(self) -> bool:
        return self.__batch_first
    
    @property
    def feed_forward(self) -> BaseFeedForward:
        return self.__feed_forward
    
    @property
    def attention(self) -> torch.nn.MultiHeadAttention:
        return self.__attention

    def forward(self, batch: FPTensor, mask: FPTensor) -> FPTensor:
        pass
