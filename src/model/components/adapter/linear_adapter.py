import torch
import typing

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.model.utils.typing import FPTensor

from src.model.components.adapter.base_adapter import BaseAdapter

class LinearAdapter(BaseAdapter):
    def __init__(self,
                 question_dim: int, 
                 common_dim: typing.Optional[int] = None) -> None:
        super().__init__(
            common_dim = common_dim,
            question_dim = question_dim,
        )

        self.__linear = self.__make_linear()

    def __make_linear(self) -> torch.nn.Linear:
        return torch.nn.Linear(
            in_features = self.question_dim,
            out_features = self.common_dim,
        )

    @property
    def linear(self) -> torch.nn.Linear:
        return self.__linear 

    def _forward(self, sequences: FPTensor) -> FPTensor:
        return self.linear(sequences)
    
import logging
import unittest

log = logging.getLogger(__name__)

class TestLinearAdapter(unittest.TestCase):
    def check_linear(self, **kwargs) -> None:
        adapter = LinearAdapter(
            common_dim = kwargs["common_dim"],
            question_dim = kwargs["question_dim"]
        )
        batch = self.generate_batch(**kwargs)
        result = adapter.forward(batch)

        gtr_size: typing.Tuple[int] = (kwargs["batch_size"], \
            kwargs["seq_len"], kwargs["common_dim"])

        assert result.size() == gtr_size

    def test_linear(self) -> None:
        log.info("Testing LinearAdapter")
        sizes = self.generate_sizes()

        for s in sizes:
            self.check_linear(**s)

    def make_generator(self, **kwargs) -> torch.Generator:
        generator = torch.Generator()
        seed = sum(kwargs.values()) + 3
        generator.manual_seed(seed)
        return generator

    def generate_batch(self, **kwargs) -> torch.Tensor:
        generator = self.make_generator(**kwargs)
        return torch.rand((kwargs["batch_size"], kwargs["seq_len"], 
                          kwargs["question_dim"]), generator = generator)

    def generate_sizes(self) -> typing.List[typing.Dict[str, int]]:
        return [
            {"batch_size": 1, "seq_len": 1, "common_dim": 1, "question_dim": 1},
            {"batch_size": 7, "seq_len": 1, "common_dim": 1, "question_dim": 2},
            {"batch_size": 1, "seq_len": 7, "common_dim": 1, "question_dim": 3},
            {"batch_size": 1, "seq_len": 1, "common_dim": 7, "question_dim": 4},
            {"batch_size": 2, "seq_len": 3, "common_dim": 5, "question_dim": 5},
        ]

if __name__ == "__main__":
    unittest.main()
