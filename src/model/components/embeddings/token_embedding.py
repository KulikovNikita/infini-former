import torch

import typing

from src.model.utils.builders import BaseBuilder
from src.model.utils.typing import FPTensor, IndexTensor

from src.model.components.embeddings.base_embedding import BaseEmbedding

class TokenEmbedding(BaseEmbedding):
    Batch = typing.Dict[str, torch.Tensor]

    def __init__(self,
                 feature_name: str,
                 embedding_count: int, 
                 embedding_dim: int, **kwargs) -> None:
        super().__init__(embedding_dim)

        self.__feature_name = feature_name
        self.__embedding_count = embedding_count
        self.__implementation = torch.nn.Embedding(
            num_embedding = self.embedding_count,
            embedding_dim = self.embedding_dim,
            **kwargs
        )

    @property
    def feature_name(self) -> str:
        return self.__feature_name

    @property
    def embedding_count(self) -> int:
        return self.__embedding_count

    @property
    def implementation(self) -> torch.nn.Embedding:
        return self.__implementation

    def _forward(self, batch: Batch, offset: int) -> FPTensor:
        tokens: IndexTensor = batch[self.feature_name]
        return self.implementation(tokens)

class TokenEmbeddingBuilder(BaseBuilder[BaseEmbedding]):
    feature_name: str
    embedding_count: int
    embedding_dim: int
    kwargs: typing.Dict[str, typing.Any] = {}

    def build(self) -> BaseEmbedding:
        return TokenEmbedding(
            embedding_count = self.embedding_count,
            embedding_dim = self.embedding_dim,
            feature_name = self.feature_name,
            **self.kwargs,
        )
