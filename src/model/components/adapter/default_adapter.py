import torch
import typing

import dataclasses

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.model.utils.typing import FPTensor
from src.model.utils.builders import BaseBuilder

from src.model.components.adapter.base_adapter import BaseAdapter
from src.model.components.adapter.linear_adapter import LinearAdapter
from src.model.components.adapter.trivial_adapter import TrivialAdapter

class DefaultAdapter(BaseAdapter):
    def __init__(self, 
                 question_dim: int, 
                 common_dim: typing.Optional[int] = None) -> None:
        super().__init__(
            common_dim = common_dim,
            question_dim = question_dim,
        )

        self.__implementation = self.__make_implementation()

        assert self.common_dim == self.implementation.common_dim
        assert self.question_dim == self.implementation.question_dim

    def _forward(self, sequences: FPTensor) -> FPTensor:
        return self.implementation(sequences)

    @property
    def implementation(self) -> BaseAdapter:
        return self.__implementation
    
    def __make_implementation(self) -> BaseAdapter:
        if self.question_dim == self.common_dim:
            cls = TrivialAdapter
        else:
            cls = LinearAdapter
        return cls(
            common_dim = self.common_dim,
            question_dim = self.question_dim, 
        )
    
@dataclasses.dataclass
class DefaultAdapterBuilder(BaseBuilder[DefaultAdapter]):
    question_dim: int
    common_dim: typing.Optional[int] = None

    def build(self) -> DefaultAdapter:
        return DefaultAdapter(
            common_dim = self.common_dim,
            question_dim = self.question_dim,
        )

    
import logging
import unittest

log = logging.getLogger(__name__)

class TestDefaultAdapter(unittest.TestCase):
    def check_default(self, **kwargs) -> None:
        adapter = DefaultAdapter(
            common_dim = kwargs["common_dim"],
            question_dim = kwargs["question_dim"]
        )
        batch = self.generate_batch(**kwargs)
        result = adapter.forward(batch)

        gtr_size: typing.Tuple[int] = (kwargs["batch_size"], \
            kwargs["seq_len"], kwargs["common_dim"])

        assert result.size() == gtr_size

    def test_default(self) -> None:
        log.info("Testing DefaultAdapter")
        sizes = self.generate_sizes()

        for s in sizes:
            self.check_default(**s)

    def make_generator(self, **kwargs) -> torch.Generator:
        generator = torch.Generator()
        seed = sum(kwargs.values()) + 5
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
