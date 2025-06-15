# SPDX-FileCopyrightText: © 2025 Authors of the roar-net-api-py project <https://github.com/roar-net/roar-net-api-py/blob/main/AUTHORS>
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from itertools import repeat
from operator import add
from typing import Self

_Value = int | float


class Lexicographic:
    def __init__(self, v: Sequence[_Value]):
        self._v = tuple(v)

    @classmethod
    def repeat(cls, v: _Value, n: int) -> Self:
        return cls(tuple(repeat(v, n)))

    @classmethod
    def zero(cls, n: int) -> Self:
        return cls.repeat(0, n)

    def __str__(self) -> str:
        return str(self._v)

    def __repr__(self) -> str:
        return f"Lexicographic({repr(self._v)})"

    def __lt__(self, other: Self) -> bool:
        return self._v < other._v

    def __le__(self, other: Self) -> bool:
        return self._v <= other._v

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Lexicographic):
            return self._v == other._v
        raise TypeError("Invalid type for Lexicographic __eq__")

    def __add__(self, other: Self) -> "Lexicographic":
        return Lexicographic(tuple(map(add, self._v, other._v)))
