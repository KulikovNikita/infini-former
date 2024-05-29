import torch

import typing

import dataclasses

from src.model.utils.typing import FPTensor, IndexTensor
from src.model.utils.builders import value_or_build, BaseBuilder

from src.model.components.embeddings.base_embedding import (
    BaseEmbedding,
)
from src.model.components.positional.log_positional import (
    LogPositional, MaybeLogPositionalBuilder,
)

class LogPositionalEmbedding(BaseEmbedding):
    Batch = typing.Mapping[str, torch.Tensor]
    def __init__(self, 
                 feature_name: str,
                 log_positional: MaybeLogPositionalBuilder) -> None:
        instance = value_or_build(log_positional)

        super().__init__(instance.digits)

        self.__log_positional = instance
        self.__feature_name = feature_name

    @property
    def feature_name(self) -> str:
        return self.__feature_name
    
    @property
    def log_positional(self) -> LogPositional:
        return self.__log_positional
    
    def forward_indicess(self, indices: IndexTensor) -> FPTensor:
        indices_size = indices.size()
        result = self.log_positional(indices)
        target_size = (*indices_size, self.embedding_dim) 
        assert result.size() == target_size
        return result
    
    def _forward(self, batch: Batch, offset: int) -> FPTensor:
        indices: IndexTensor = batch[self.feature_name]
        result = self.forward_indicess(indices)
        self.__validate_output(result)
        return result

@dataclasses.dataclass
class LogPositionalEmbeddingBuilder(BaseBuilder[BaseEmbedding]):
    feature_name: str
    log_positional: MaybeLogPositionalBuilder

    def build(self) -> BaseEmbedding:
        return LogPositionalEmbedding(
            feature_name = self.feature_name,
            log_positional = self.log_positional,
        )
