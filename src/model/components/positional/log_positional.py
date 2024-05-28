import torch

import typing

import dataclasses

from src.model.utils.typing import IndexTensor, FPTensor
from src.model.utils.builders import BaseBuilder, MaybeBuilder
    
class LogPositional(torch.nn.Module):
    def __init__(self, digits: int, base: float = 2.0,
                 dtype: torch.dtype = torch.float32,) -> None:
        super().__init__()

        self.__base = base
        self.__dtype = dtype
        self.__digits = digits

        self.__powers = LogPositional.__make_powers(
            digits = digits, base = base, 
            dtype = dtype
        )
    
    @staticmethod
    def __make_powers(digits, base, dtype) -> FPTensor:
        digits_range = torch.arange(0, digits, dtype = torch.int32)

        if base == 2.0:
            powers = torch.exp2(digits_range)
        else:
            base_tensor = torch.Tensor(base, dtype = dtype)
            powers = torch.pow(base_tensor, digits_range)
        powers = powers.to(dtype)

        assert len(powers) == digits
        assert powers.dtype == dtype
        assert not powers.requires_grad

        return powers

    @property
    def base(self) -> int:
        return self.__base

    @property
    def digits(self) -> int:
        return self.__digits
    
    @property
    def dtype(self) -> torch.dtype:
        return self.__dtype    

    def forward(self, positions: IndexTensor) -> FPTensor:
        powered = positions.unsqueeze(-1) * self.__powers
        result = torch.special.expit(powered)

        target_size = (*positions.size(), self.digits)
        assert result.size() == target_size
        assert result.dtype == self.dtype

        return result
    
@dataclasses.dataclass
class LogPositionalBuilder(BaseBuilder[LogPositional]):
    digits: int
    base: float = 2.0
    dtype: torch.dtype = torch.float32

    def build(self) -> LogPositional:
        return LogPositional(
            digits = self.digits,
            dtype = self.dtype,
            base = self.base,
        )
    
MaybeLogPositionalBuilder = MaybeBuilder[LogPositional]

import unittest

class TestLogPositional(unittest.TestCase):
    def _make_generator(self, seed = 777):
        gen = torch.Generator()
        return gen.manual_seed(seed)
    
    def _dummy_positions(self, size):
        seed = 777 + sum(size)
        gen = self._make_generator(seed)
        return torch.randint(
            low = 0,
            high = seed,
            size = size,
            generator = gen,
            dtype = torch.int32,
        )

    def assert_sizes(self, size, digits):
        target_size = (*size, digits)
        positions = self._dummy_positions(size)
        encoder = LogPositional(digits)
        encoded = encoder(positions)

        self.assertEqual(encoded.size(), target_size)

    def test_sizes(self):
        sizes = [
            (1,), (5,), (1, 2),
            (2, 5), (3, 12, 4),
        ]

        digits = [1, 2, 15, 32]

        for s in sizes:
            for d in digits:
                self.assert_sizes(s, d)

if __name__ == "__main__":
    unittest.main()
