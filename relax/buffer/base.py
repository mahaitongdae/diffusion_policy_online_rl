from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Generic, Tuple, TypeVar, TYPE_CHECKING

T = TypeVar("T")

if TYPE_CHECKING:
    import numpy as np

class Buffer(Generic[T], metaclass=ABCMeta):
    @abstractmethod
    def add(self, sample: T, *, from_jax: bool = False) -> None:
        ...

    @abstractmethod
    def add_batch(self, samples: T, *, from_jax: bool = False) -> None:
        ...

    @abstractmethod
    def sample(self, size: int, *, to_jax: bool = False) -> T:
        ...

    @abstractmethod
    def sample_with_indices(self, size: int, *, to_jax: bool = False) -> Tuple[T, "np.ndarray"]:
        ...

    @abstractmethod
    def replace(self, indices: "np.ndarray", samples: T, *, from_jax: bool = False) -> None:
        ...

    @abstractmethod
    def save(self, path: Path) -> None:
        ...
