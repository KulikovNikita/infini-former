import torch

import typing

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.model.utils.builders import MaybeBuilder, BaseBuilder

class AttentionMasking(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, size: typing.Tuple[int, int, int]) -> torch.BoolTensor:
        _, seq_len, _  = size
        ones_size = (seq_len, seq_len)
        ones = torch.ones(*ones_size, dtype = torch.bool)
        result = ~torch.tril(ones, diagonal = -1)
        assert result.size() == ones_size
        return result

class AttentionMaskingBuilder(BaseBuilder[AttentionMasking]):
    def build(self) -> AttentionMasking:
        return AttentionMasking()
    
MaybeAttentionMaskingBuilder = MaybeBuilder[AttentionMasking]

DEFAULT_ATTENTION_MASKING: AttentionMasking = AttentionMasking()

import logging
import unittest

class TestMasking(unittest.TestCase):
    def test_masking_exemplar(self) -> None:
        groundtruth = torch.Tensor([
            [1, 1, 1], 
            [0, 1, 1],
            [0, 0, 1],
        ]).to(dtype = torch.bool)
        
        masking = AttentionMasking()
        result = masking(size = (1, 3, 1))

        self.assertTrue(result.size() == (3, 3))
        self.assertTrue(torch.all(result == groundtruth))

if __name__ == "__main__":
    unittest.main()
