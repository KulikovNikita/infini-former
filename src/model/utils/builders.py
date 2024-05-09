import abc
import typing

BuildType = typing.TypeVar("BuildType")

class BaseBuilder[BuildType]:
    @abc.abstractmethod
    def build(self) -> BuildType:
        pass 

MaybeBuilder = typing.Union[BuildType, BaseBuilder[BuildType]]

def is_builder(maybe_builder: MaybeBuilder) -> bool:
    return isinstance(maybe_builder, BaseBuilder)

def value_or_build(maybe_builder: MaybeBuilder[BuildType]) -> BuildType:
    return maybe_builder.build() if is_builder(maybe_builder) else maybe_builder 

