import abc

import torch

import typing

from src.model.utils.typing import FPTensor
from src.model.utils.builders import MaybeBuilder

class BaseEmbedding(torch.nn.Module):
    Batch = typing.Dict[str, torch.Tensor]

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()

        self.__embedding_dim = embedding_dim

    @property
    def embedding_dim(self) -> int:
        return self.__embedding_dim

    def __validate_output(self, output: FPTensor) -> None:
        _, _, embedding_dim = output.size()
        assert embedding_dim == self.embedding_dim

    @abc.abstractmethod
    def _forward(self, batch: Batch, offset: int) -> FPTensor:
        pass

    def forward(self, batch: Batch, offset: int = 0) -> FPTensor:
        output: FPTensor = self._forward(batch, offset)
        self.__validate_output(output, offset)
        return output

MaybeEmbeddingBuilder = MaybeBuilder[BaseEmbedding]
