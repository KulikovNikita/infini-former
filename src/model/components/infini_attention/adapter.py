import abc
import typing

import torch

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.model.utils.typing import value_or_default, FPTensor

class BaseAdapter(torch.nn.Module):
    def __init__(self, 
                 question_dim: int, 
                 common_dim: typing.Optional[int] = None) -> None:
        self.__question_dim = question_dim
        self.__common_dim = value_or_default(common_dim, question_dim)
        
    @property
    def common_dim(self) -> int:
        return self.__common_dim
    @property
    def question_dim(self) -> int:
        return self.__question_dim

    @abc.abstractmethod
    def _forward(self, sequences: FPTensor) -> FPTensor:
        pass

    def __check_input(self, sequences: FPTensor) -> None:
        assert self.question_dim == sequences.size(-1)

    def __check_output(self, sequences: FPTensor, output: FPTensor) -> None:
        seq_size = list(sequences.size())
        correct_size = (*seq_size[:-1], self.common_dim)
        assert correct_size == output.size()

    def forward(self, sequences: FPTensor) -> FPTensor:
        self.__check_input(sequences)
        result = self._forward(sequences)
        self.__check_output(sequences, result)
        return result

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

class NonTrivialAdapter(BaseAdapter):
    def __init__(self,
                 question_dim: int, 
                 common_dim: typing.Optional[int] = None) -> None:
        super().__init__(
            common_dim = common_dim,
            question_dim = question_dim,
        )

    def linear(self) -> torch.nn.Linear:
        return self.__linear 

    def _forward(self, sequences: FPTensor) -> FPTensor:
        return self.linear(sequences)
        
class DefaultAdapter(BaseAdapter):
    def __init__(self, 
                 question_dim: int, 
                 common_dim: typing.Optional[int] = None) -> None:
        super().__init__(question_dim, common_dim)

        self.__implementation = self.__make_implementation()

        assert self.common_dim == self.implementation.common_dim
        assert self.question_dim == self.implementation.question_dim

    def _forward(self, sequences: FPTensor) -> FPTensor:
        return self.implementation(sequences)

    @property
    def implementation(self) -> BaseAdapter:
        return self.__implementation
    
    def __make_implementation(self) -> BaseAdapter:
        if self.question_dim == self.query_dim:
            cls = TrivialAdapter
        else:
            cls = NonTrivialAdapter
        return cls(
            common_dim = self.common_dim,
            question_dim = self.question_dim, 
        )

import logging
import unittest

log = logging.getLogger(__name__)

#class AdapterFixture:
#    def 

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
