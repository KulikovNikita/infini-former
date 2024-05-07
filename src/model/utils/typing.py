import torch

import typing

IndexTensor = typing.Union[torch.IntTensor, torch.LongTensor]

FPTensor = typing.Union[torch.FloatTensor, torch.DoubleTensor]

_T = typing.TypeVar("_T")

def value_or_default(optional: typing.Optional[_T], default: _T) -> _T:
    return default if optional is None else optional
