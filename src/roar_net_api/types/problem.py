# SPDX-FileCopyrightText: © 2025 Authors of the roar-net-api-py project <https://github.com/roar-net/roar-net-api-py/blob/main/AUTHORS>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Protocol, TypeVar

from ..operations import (
    SupportsConstructionNeighbourhood,
    SupportsDestructionNeighbourhood,
    SupportsEmptySolution,
    SupportsHeuristicSolution,
    SupportsLocalNeighbourhood,
    SupportsRandomSolution,
)

_TConstructiveNeighbourhood = TypeVar("_TConstructiveNeighbourhood", covariant=True)
_TLocalNeighbourhood = TypeVar("_TLocalNeighbourhood", covariant=True)
_TSolution = TypeVar("_TSolution", covariant=True)


class Problem(
    SupportsConstructionNeighbourhood[_TConstructiveNeighbourhood],
    SupportsDestructionNeighbourhood[_TLocalNeighbourhood],
    SupportsEmptySolution[_TSolution],
    SupportsHeuristicSolution[_TSolution],
    SupportsLocalNeighbourhood[_TSolution],
    SupportsRandomSolution[_TSolution],
    Protocol,
): ...
