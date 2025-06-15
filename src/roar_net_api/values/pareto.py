# SPDX-FileCopyrightText: © 2025 Authors of the roar-net-api-py project <https://github.com/roar-net/roar-net-api-py/blob/main/AUTHORS>
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable, Sequence
from itertools import repeat
from operator import add, eq, le
from typing import Generic, Self, TypeVar

from .float import Float

_T = TypeVar("_T", int, float, Float)


def _pareto_lt(a: Iterable[_T], b: Iterable[_T]) -> bool:
    eq = True
    for x, y in zip(a, b):
        if y < x:
            return False
        eq = eq and x == y
    return not eq


def _pareto_le(a: Iterable[_T], b: Iterable[_T]) -> bool:
    return all(map(le, a, b))


def _pareto_eq(a: Iterable[_T], b: Iterable[_T]) -> bool:
    return all(map(eq, a, b))


class Pareto(Generic[_T]):
    def __init__(self, v: Sequence[_T]):
        self._v: tuple[_T, ...] = tuple(v)

    @classmethod
    def repeat(cls, v: _T, n: int) -> Self:
        return cls(tuple(repeat(v, n)))

    def __str__(self) -> str:
        values = ", ".join(map(str, self._v))
        return f"({values})"

    def __repr__(self) -> str:
        return f"Pareto({repr(self._v)})"

    def __lt__(self, other: Self) -> bool:
        return _pareto_lt(self._v, other._v)

    def __le__(self, other: Self) -> bool:
        return _pareto_le(self._v, other._v)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, type(self)):
            return _pareto_eq(self._v, other._v)
        raise TypeError("Invalid type for Pareto __eq__")

    def __add__(self, other: Self) -> Self:
        return type(self)(tuple(map(add, self._v, other._v)))

    def __getitem__(self, index: int) -> _T:
        return self._v[index]
