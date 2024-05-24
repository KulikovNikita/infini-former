import abc
import typing

import torch

from src.model.utils.typing import value_or_default, FPTensor
from src.model.utils.builders import BaseBuilder, MaybeBuilder

class BaseAdapter(torch.nn.Module):
    def __init__(self, 
                 question_dim: int, 
                 common_dim: typing.Optional[int] = None) -> None:
        super().__init__()
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
    
AdapterBuilder = BaseBuilder[BaseAdapter]
MaybeAdapter = typing.Optional[BaseAdapter]
MaybeAdapterBuilder = MaybeBuilder[BaseAdapter]
