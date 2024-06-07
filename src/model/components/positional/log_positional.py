import torch

import dataclasses

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.model.utils.typing import IndexTensor, FPTensor
from src.model.utils.builders import BaseBuilder, MaybeBuilder
    
class LogPositional(torch.nn.Module):
    def __init__(self, digits: int, base: float = 2.0,
                 dtype: torch.dtype = torch.float32,) -> None:
        super().__init__()

        self.__base = base
        self.__dtype = dtype
        self.__digits = digits

        self.__direct_powers, self.__inverse_powers = LogPositional.__make_powers(
            digits = digits, base = base, 
            dtype = dtype
        )

    @staticmethod
    def __check_powers(digits, dtype, powers) -> None:
        assert len(powers) == digits
        assert powers.dtype == dtype
        assert not powers.requires_grad
    
    @staticmethod
    def __make_powers(digits, base, dtype) -> FPTensor:
        digits_range = torch.arange(0, digits, dtype = torch.int32)

        if base == 2.0:
            direct_powers = torch.exp2(-digits_range)
            inverse_powers = torch.exp2(+digits_range)
        else:
            base_tensor = torch.Tensor(base, dtype = dtype)
            direct_powers = torch.pow(base_tensor, -digits_range)
            inverse_powers = torch.pow(base_tensor, +digits_range)
        direct_powers = direct_powers.to(dtype)
        inverse_powers = inverse_powers.to(dtype)

        LogPositional.__check_powers(digits, dtype, direct_powers)
        LogPositional.__check_powers(digits, dtype, inverse_powers)

        return direct_powers, inverse_powers

    @property
    def base(self) -> int:
        return self.__base

    @property
    def digits(self) -> int:
        return self.__digits
    
    @property
    def dtype(self) -> torch.dtype:
        return self.__dtype    
    
    @property
    def direct_powers(self) -> FPTensor:
        return self.__direct_powers
    
    @property
    def inverse_powers(self) -> FPTensor:
        return self.__inverse_powers
    
    def forward(self, positions: IndexTensor) -> FPTensor:
        return self.encode(positions = positions)

    def encode(self, positions: IndexTensor) -> FPTensor:
        powered = positions.unsqueeze(-1) * self.direct_powers
        result = torch.exp2(powered)

        target_size = (*positions.size(), self.digits)
        assert result.size() == target_size
        assert result.dtype == self.dtype

        return result
    
    def decode(self, embeddings: FPTensor) -> FPTensor:
        unexped = torch.log2(embeddings)
        unpowered = unexped * self.inverse_powers
        result, _ = torch.median(unpowered, dim = -1)

        result_size = list(result.size())
        target_size = list(embeddings.size())
        assert result_size == target_size[:-1]
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
        gen.manual_seed(seed)
        return gen
    
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

    def assert_decode(self, size, digits):
        positional = LogPositional(digits)
        positions = self._dummy_positions(size)
        encoded = positional.encode(positions)
        decoded = positional.decode(encoded)
        casted = positions.to(decoded.dtype)

        condition = torch.allclose(decoded, casted)
        self.assertTrue(condition)

    def test_sizes(self):
        sizes = [
            (1,), (5,), (1, 2),
            (2, 5), (3, 12, 4),
        ]

        digits = [7, 8, 15, 17, 32]

        for s in sizes:
            for d in digits:
                self.assert_sizes(s, d)
                self.assert_decode(s, d)

if __name__ == "__main__":
    unittest.main()
