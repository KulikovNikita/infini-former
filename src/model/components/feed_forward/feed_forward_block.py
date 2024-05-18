import torch

import dataclasses

from src.model.utils.typing import FPTensor
from src.model.utils.builders import value_or_build, BaseBuilder, MaybeBuilder

from src.model.components.activation.activations import DEFAULT_ACTIVATION
from src.model.components.activation.base_activation import (
    BaseActivation, MaybeActivationBuilder
)

class FeedForwardBlock(torch.nn.Module):
    def __init__(self, 
                 hidden_size: int,
                 dropout: float = 0.3,
                 activation: MaybeActivationBuilder = DEFAULT_ACTIVATION) -> None:
        super().__init__()
        
        self.__dropout = dropout
        self.__hidden_size = hidden_size
        self.__activation = value_or_build(activation)

        self.__dropout_layer = self.__make_dropout_layer()
        self.__convolution_layer = self.__make_convolution_layer()

    @property
    def hidden_size(self) -> int:
        return self.__hidden_size

    @property
    def activation(self) -> BaseActivation:
        return self.__activation
    
    @property
    def dropout(self) -> float:
        return self.__dropout
    
    def __make_dropout_layer(self) -> torch.nn.Dropout:
        return torch.nn.Dropout(p = self.dropout)
    
    @property
    def dropout_layer(self) -> torch.nn.Dropout:
        return self.__dropout_layer
    
    def __make_convolution_layer(self) -> torch.nn.Conv1d:
        return torch.nn.Conv1d(
            kernel_size = 1,
            in_channels = self.hidden_size,
            out_channels = self.hidden_size,
        )

    @property
    def convolution_layer(self) -> torch.nn.Conv1d:
        return self.__convolution_layer
    
    def forward_wo_activation(self, sequences: FPTensor) -> FPTensor:
        sequences_size = sequences.size()
        
        sequences_t = torch.transpose(sequences, 1, 2)
        convolved = self.convolution_layer(sequences_t)
        dropped_t = torch.transpose(convolved, 1, 2)
        dropped_out = self.dropout_layer(dropped_t)

        assert dropped_out.size() == sequences_size
        return dropped_out

    def forward(self, sequences: FPTensor) -> FPTensor:
        sequences_size = sequences.size()

        dropped_out = self.forward_wo_activation(sequences)
        activated = self.activation(dropped_out)

        assert activated.size() == sequences_size
        return activated

@dataclasses.dataclass
class FeedForwardBlockBuilder(BaseBuilder[FeedForwardBlock]):
    hidden_size: int
    dropout: float = 0.3
    activation: MaybeActivationBuilder = DEFAULT_ACTIVATION

    def build(self) -> FeedForwardBlock:
        return FeedForwardBlock(
            dropout = self.dropout,
            activation = self.activation,
            hidden_size = self.hidden_size,
        )
    
MaybeFFBlockBuilder = MaybeBuilder[FeedForwardBlock]
