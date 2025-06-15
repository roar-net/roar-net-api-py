# SPDX-FileCopyrightText: © 2025 Authors of the roar-net-api-py project <https://github.com/roar-net/roar-net-api-py/blob/main/AUTHORS>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Protocol, Self, runtime_checkable
from math import isclose


@runtime_checkable
class _SupportsFloat(Protocol):
    def __float__(self) -> float: ...


class Float:
    def __init__(self, v: float, atol: float, rtol: float):
        self._v = float(v)
        self._atol = atol
        self._rtol = rtol

    @classmethod
    def zero(cls, atol: float = 0.0, rtol: float = 0.0) -> Self:
        return cls(0.0, atol, rtol)

    def __str__(self) -> str:
        return str(self._v)

    def __repr__(self) -> str:
        return f"Float({repr(self._v)}, {repr(self._atol)}, {repr(self._rtol)})"

    def __lt__(self, other: _SupportsFloat) -> bool:
        return self._v < float(other) and not isclose(self._v, other, abs_tol=self._atol, rel_tol=self._rtol)

    def __le__(self, other: _SupportsFloat) -> bool:
        return self._v < float(other) or isclose(self._v, other, abs_tol=self._atol, rel_tol=self._rtol)

    def __eq__(self, other: object) -> bool:
        assert isinstance(other, _SupportsFloat)
        return isclose(self._v, other, abs_tol=self._atol, rel_tol=self._rtol)

    def __add__(self, other: Self) -> "Float":
        return Float(self._v + other._v, self._atol, self._rtol)

    def __float__(self) -> float:
        return self._v
