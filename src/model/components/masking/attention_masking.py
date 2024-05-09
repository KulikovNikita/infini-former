import torch

import typing

class AttentionMasking(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, size: typing.Tuple[int, int, int]) -> torch.BoolTensor:
        _, seq_len, _  = size
        ones_size = (seq_len, seq_len)
        ones = torch.ones(*ones_size, dtype = torch.bool)
        triagonal = torch.tril(ones, diagonal = -1)
        result = (~triagonal)[None, :, :]
        assert result.size() == (1, *ones_size)
        return result

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

        self.assertTrue(result.size() == (1, 3, 3))
        self.assertTrue(torch.all(result == groundtruth))

if __name__ == "__main__":
    unittest.main()
