import torch

import typing

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.model.utils.builders import BaseBuilder
from src.model.utils.typing import FPTensor, IndexTensor

from src.model.components.embeddings.base_embedding import BaseEmbedding

def get_logits(sequences: FPTensor, embeddings: FPTensor, normalize: bool = True) -> FPTensor:
    batch_size, seq_len, embed_dim = sequences.size()
    embed_count, groundtruth_dim = embeddings.size()
    assert embed_dim == groundtruth_dim

    if normalize:
        seq_proc = torch.nn.functional.normalize(sequences, dim = -1)
        emb_proc = torch.nn.functional.normalize(embeddings, dim = -1)
    else:
        seq_proc, emb_proc = sequences, embeddings.clone()

    tiling: typing.Tuple = (batch_size, 1, 1)
    emb_proc_3d: FPTensor = emb_proc[None, :, :]
    tiled_embeddings = torch.tile(emb_proc_3d, tiling)
    result = torch.bmm(seq_proc, tiled_embeddings.mT)

    target_size = (batch_size, seq_len, embed_count) 
    assert target_size == result.size()
    return result

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
            num_embeddings = self.embedding_count,
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
        return self.forward_tokens(tokens)
    
    def forward_tokens(self, tokens: IndexTensor) -> FPTensor:
        batch_size, seq_len = tokens.size()
        result = self.implementation(tokens)
        target = (batch_size, seq_len, self.embedding_dim)
        assert target == result.size()
        return result
    
    def get_logits(self, sequences: FPTensor, normalize: bool = True) -> FPTensor:
        weights: FPTensor = self.implementation.weight
        return get_logits(sequences, weights, normalize)
    
    def set_weight(self, embeddings: FPTensor) -> "TokenEmbedding":
        target_size = (self.embedding_count, self.embedding_dim)
        assert target_size == embeddings.size()

        parameter = torch.nn.Parameter(embeddings)
        self.__implementation.weight = parameter

        return self

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

import logging
import unittest

log = logging.getLogger(__name__)

class TestTokenEmbedding(unittest.TestCase):
    def make_gen(self, **kwargs) -> torch.Generator:
        seed = sum(kwargs.values())**2 + 777
        generator = torch.Generator()
        generator.manual_seed(seed)
        torch.manual_seed(seed)
        return generator

    def gen_embeddings(self, gen, **kwargs) -> FPTensor:
        size: typing.Tuple = (kwargs["ec"], kwargs["ed"])
        return torch.randn(size, generator = gen)
    
    def make_prototype(self, **kwargs) -> TokenEmbedding:
        return TokenEmbedding(
            embedding_count = kwargs["ec"],
            embedding_dim = kwargs["ed"],
            feature_name = "tokens",
        )
    
    def make_embeddings(self, gen, **kwargs) -> TokenEmbedding:
        proto = self.make_prototype(**kwargs)
        embed = self.gen_embeddings(gen, **kwargs)
        return proto.set_weight(embed)
    
    def gen_batch(self, gen, **kwargs) -> FPTensor:
        size: typing.Tuple = (kwargs["bs"], kwargs["sl"])
        return torch.randint(0, kwargs["ec"], size, generator = gen)
    
    def check_similarities(self, embs: TokenEmbedding, seqs: IndexTensor, vecs: FPTensor) -> None:
        v_bs, v_sl, v_emb_dim = vecs.size()
        e_emb_dim = embs.embedding_dim
        s_bs, s_sl = seqs.size() 

        self.assertEqual((v_bs, v_sl), (s_bs, s_sl))
        self.assertEqual(e_emb_dim, v_emb_dim)

        logits = embs.get_logits(vecs)
        _, tokens = torch.topk(logits, k = 1, dim = -1)
        tokens_2d: IndexTensor = tokens.squeeze(-1)

        self.assertTrue(torch.all(tokens_2d == seqs))
    
    def check_size(self, **kwargs) -> None:
        gen = self.make_gen(**kwargs)
        seqs = self.gen_batch(gen, **kwargs)
        embs = self.make_embeddings(gen, **kwargs)
        vecs = embs({"tokens": seqs})
        self.check_similarities(embs, seqs, vecs)

    def test_token_embedding(self) -> None:
        for sizes in self.get_sizes():
            self.check_size(**sizes)

    def get_sizes(self) -> typing.List[typing.Dict[str, int]]:
        return [
            {"ec": 1, "ed": 1, "bs": 1, "sl": 1},
            {"ec": 2, "ed": 1, "bs": 1, "sl": 1},
            {"ec": 1, "ed": 2, "bs": 1, "sl": 1},
            {"ec": 1, "ed": 1, "bs": 2, "sl": 1},
            {"ec": 1, "ed": 1, "bs": 1, "sl": 2},
            {"ec": 2, "ed": 3, "bs": 4, "sl": 5},
            {"ec": 3, "ed": 4, "bs": 5, "sl": 6},
            {"ec": 4, "ed": 5, "bs": 6, "sl": 7},
            {"ec": 5, "ed": 6, "bs": 7, "sl": 8},
            {"ec": 6, "ed": 7, "bs": 8, "sl": 9},
        ]

if __name__ == "__main__":
    unittest.main()
