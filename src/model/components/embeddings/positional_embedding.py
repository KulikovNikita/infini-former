import torch

import typing

from src.model.utils.typing import FPTensor, IndexTensor
from src.model.utils.builders import value_or_build, BaseBuilder

from src.model.components.embeddings.base_embedding import (
    BaseEmbedding,
)
from src.model.components.positional.log_positional import (
    LogPositional, MaybeLogPositionalBuilder,
)

class LogPositionalEmbedding(BaseEmbedding):
    def __init__(self, 
                 feature_name: str,
                 embedding_dim: int,
                 log_positional: MaybeLogPositionalBuilder) -> None:
        super().__init__(embedding_dim)

        self.__feature_name = feature_name
        self.__log_positional = value_or_build(log_positional)

    @property
    def feature_name(self) -> str:
        return self.__feature_name
    
    @property
    def log_positional(self) -> LogPositional:
        return self.__log_positional
    
    def _forward(self, batch: Batch, offset: int) -> FPTensor:
        indices: IndexTensor = batch[self.feature_name]
        
        return 

