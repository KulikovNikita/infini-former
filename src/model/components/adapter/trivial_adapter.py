import torch
import typing

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.model.utils.typing import FPTensor

from src.model.components.adapter.base_adapter import BaseAdapter

class TrivialAdapter(BaseAdapter):
    def __init__(self,
                 question_dim: int, 
                 common_dim: typing.Optional[int] = None) -> None:
        super().__init__(
            common_dim = common_dim,
            question_dim = question_dim,
        )

        assert self.common_dim == self.question_dim

    def _forward(self, sequences: FPTensor) -> FPTensor:
        return sequences
    
import logging
import unittest

log = logging.getLogger(__name__)

class TestTrivialAdapter(unittest.TestCase):
    def check_trivial(self, **kwargs) -> None:
        adapter = TrivialAdapter(
            common_dim = kwargs["common_dim"],
            question_dim = kwargs["question_dim"]
        )
        batch = self.generate_batch(**kwargs)
        result = adapter.forward(batch)

        assert torch.allclose(batch, result)

    def test_trivial(self) -> None:
        log.info("Testing TrivialAdapter")
        sizes = self.generate_sizes()

        for s in sizes:
            self.check_trivial(**s)

    def make_generator(self, **kwargs) -> torch.Generator:
        generator = torch.Generator()
        seed = sum(kwargs.values()) + 7
        generator.manual_seed(seed)
        return generator

    def generate_batch(self, **kwargs) -> torch.Tensor:
        generator = self.make_generator(**kwargs)
        return torch.rand((kwargs["batch_size"], kwargs["seq_len"], 
                          kwargs["question_dim"]), generator = generator)

    def generate_sizes(self) -> typing.List[typing.Dict[str, int]]:
        return [
            {"batch_size": 1, "seq_len": 1, "common_dim": 1, "question_dim": 1},
            {"batch_size": 7, "seq_len": 1, "common_dim": 1, "question_dim": 1},
            {"batch_size": 1, "seq_len": 7, "common_dim": 1, "question_dim": 1},
            {"batch_size": 1, "seq_len": 1, "common_dim": 7, "question_dim": 7},
            {"batch_size": 2, "seq_len": 3, "common_dim": 5, "question_dim": 5},
        ]

if __name__ == "__main__":
    unittest.main()
