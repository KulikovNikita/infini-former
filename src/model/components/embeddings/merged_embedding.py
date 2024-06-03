import torch

import typing

import dataclasses

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.model.utils.builders import value_or_build, BaseBuilder
from src.model.utils.typing import value_or_default, FPTensor, IndexTensor

from src.model.components.embeddings.base_embedding import (
    BaseEmbedding, MaybeEmbeddingBuilder,
)

class MergedEmbedding(BaseEmbedding):
    SlicesList = typing.List[FPTensor]
    Batch = typing.Dict[str, torch.Tensor]
    EmbeddingList = typing.List[BaseEmbedding]
    MaybeEmbeddingList = typing.List[MaybeEmbeddingBuilder]

    def __init__(self, embedding_list: MaybeEmbeddingList) -> None:
        preped_embs = MergedEmbedding.__prepare_module(embedding_list)
        emb_offsets = MergedEmbedding.__compute_offsets(preped_embs)
        embedding_dim: int = emb_offsets[-1].item()

        super().__init__(embedding_dim)

        self.__embeddings, self.__offsets = preped_embs, emb_offsets

    @staticmethod
    def __prepare_list(embedding_list) -> EmbeddingList:
        return [value_or_build(emb) for emb in embedding_list]

    @staticmethod
    def __prepare_module(embedding_list) -> torch.nn.ModuleList:
        preped_embs = MergedEmbedding.__prepare_list(embedding_list)
        return torch.nn.ModuleList(preped_embs)

    @staticmethod
    def __compute_offsets(embedding_list) -> IndexTensor:
        emb_dims = [0, *[emb.embedding_dim for emb in embedding_list]]
        dim_tensor = torch.asarray(emb_dims, dtype = torch.int64)
        assert len(dim_tensor) == len(embedding_list) + 1
        return torch.cumsum(dim_tensor, -1, dtype = torch.int64)
    
    @property
    def offsets(self) -> IndexTensor:
        return self.__offsets
    
    @property
    def embeddings(self) -> torch.nn.ModuleList:
        return self.__embeddings
    
    def forward_to_list(self, batch: Batch) -> SlicesList:
        rolling_offset: int = 0
        slices_list: SlicesList = []
        for i, emb in enumerate(self.embeddings):
            assert rolling_offset == self.offsets[i].item()
            emb_values: FPTensor = emb.forward(batch)
            rolling_offset += emb.embedding_dim
            slices_list.append(emb_values)
        assert rolling_offset == self.embedding_dim
        return slices_list
    
    def _forward(self, batch: Batch, offset: int) -> FPTensor:
        slices_list = self.forward_to_list(batch)
        result: FPTensor = torch.cat(slices_list, dim = -1)
        assert result.size(-1) == self.embedding_dim
        return result
    
@dataclasses.dataclass
class MergedEmbeddingBuilder(BaseBuilder[MergedEmbedding]):
    embedding_list: typing.List[MaybeEmbeddingBuilder]

    def build(self) -> BaseEmbedding:
        return MergedEmbedding(
            embedding_list = self.embedding_list,
        )


import logging
import unittest

from src.model.components.embeddings.mock_embedding import MockEmbedding

log = logging.getLogger(__name__)

class TestMergedEmbedding(unittest.TestCase):
    def make_generator(self, **kwargs) -> torch.Generator:
        seed = sum(kwargs["emb_sizes"]) + 777
        generator = torch.Generator()
        generator.manual_seed(seed)
        return generator

    def make_embeddings(self, gen, **kwargs) -> typing.List[MockEmbedding]:
        embedding_sizes: typing.List[int] = kwargs["emb_sizes"]
        return [MockEmbedding(es, generator = gen) for es in embedding_sizes]
    
    def make_merged(self, gen, **kwargs) -> MergedEmbedding:
        embs = self.make_embeddings(gen, **kwargs)
        result = MergedEmbedding(embedding_list = embs)

        embedding_size: int = sum(kwargs["emb_sizes"])
        self.assertEqual(embedding_size, result.embedding_dim)

        return result
    
    def check_batch(self, merged, batch) -> None:
        embeddings: FPTensor = merged(batch)

        bs, sl, ed = embeddings.size()
        self.assertEqual(sl, batch["seq_len"])
        self.assertEqual(bs, batch["batch_size"])
        self.assertEqual(ed, merged.embedding_dim)
    
    def check_sizes(self, **kwargs) -> None:
        gen = self.make_generator(**kwargs)
        merged = self.make_merged(gen, **kwargs)

        for batch in self.get_batches():
            self.check_batch(merged, batch)

    def test_sizes(self) -> None:
        for sizes in self.get_sizes():
            self.check_sizes(**sizes)

    def get_batches(self):
        return [
            {"batch_size": 1, "seq_len": 1},
            {"batch_size": 2, "seq_len": 1},
            {"batch_size": 1, "seq_len": 2},
            {"batch_size": 2, "seq_len": 2},
            {"batch_size": 5, "seq_len": 7},
        ]

    def get_sizes(self):
        return [
            {"emb_sizes": [1]},
            {"emb_sizes": [1]},
            {"emb_sizes": [1]},
            {"emb_sizes": [2]},
            {"emb_sizes": [1, 2]},
            {"emb_sizes": [2, 2]},
            {"emb_sizes": [4, 5]},
            {"emb_sizes": [5, 6, 7]},
        ]

if __name__ == "__main__":
    unittest.main()

