import abc
import typing

BuildType = typing.TypeVar("BuildType")

class Builder[BuildType]:
    @abc.abstractmethod
    def build(self) -> BuildType:
        pass 

MaybeBuilder = typing.Union[BuildType, Builder[BuildType]]

def is_builder(maybe_builder: MaybeBuilder) -> bool:
    return issubclass(maybe_builder, Builder)

def value_or_build(maybe_builder: MaybeBuilder) -> BuildType:
    return maybe_builder.build() if is_builder(maybe_builder) else maybe_builder 

