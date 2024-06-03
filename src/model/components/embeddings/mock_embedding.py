import torch

import typing

import dataclasses

from src.model.utils.builders import BaseBuilder
from src.model.utils.typing import value_or_default, FPTensor

from src.model.components.embeddings.base_embedding import BaseEmbedding

class MockEmbedding(BaseEmbedding):
    Batch = typing.Dict[str, typing.Any]
    MaybeGenerator = typing.Optional[torch.Generator]
    def __init__(self, 
                 embedding_dim: int,
                 dtype = torch.float32,
                 seq_len_feature: str = "seq_len",
                 batch_size_feature: str = "batch_size",
                 generator: MaybeGenerator = None,) -> None:
        super().__init__(embedding_dim)

        self.dtype = dtype
        self.generator = value_or_default(
            generator, torch.Generator()
        )
        self.seq_len_feature = seq_len_feature
        self.batch_size_feature = batch_size_feature

    def _forward(self, batch: Batch, offset: int) -> FPTensor:
        sl: int = batch[self.seq_len_feature]
        bs: int = batch[self.batch_size_feature]
        size: typing.Tuple[int, int, int] = \
                    (bs, sl, self.embedding_dim)
        return torch.rand(
            size, generator = self.generator,
            dtype = self.dtype,
        )

@dataclasses.dataclass
class MockEmbeddingBuilder(BaseBuilder[BaseEmbedding]):
    embedding_dim: int
    seq_len_feature: str = "seq_len"
    batch_size_feature: str = "batch_size"
    generator: typing.Optional[torch.Generator] = None

    def build(self) -> BaseEmbedding:
        return MockEmbedding(
            generator = self.generator,
            embedding_dim = self.embedding_dim,
            seq_len_feature = self.seq_len_feature,
            batch_size_feature = self.batch_size_feature,
        )
