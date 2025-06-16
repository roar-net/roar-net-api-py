#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: © 2025 Authors of the roar-net-api-py project <https://github.com/roar-net/roar-net-api-py/blob/main/AUTHORS>
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable
from logging import getLogger
from typing import Protocol, TypeVar

from ..operations import (
    SupportsApplyMove,
    SupportsLocalNeighbourhood,
    SupportsObjectiveValueIncrement,
    SupportsRandomMovesWithoutReplacement,
    SupportsSubNeighbourhoods,
)

log = getLogger(__name__)

_TSolution = TypeVar("_TSolution")


class _Increment(Protocol):
    def __lt__(self, other: int) -> bool: ...


class _Move(
    SupportsApplyMove[_TSolution],
    SupportsObjectiveValueIncrement[_TSolution, _Increment],
    Protocol[_TSolution],
): ...


class _SubNeighbourhood(SupportsRandomMovesWithoutReplacement[_TSolution, _Move[_TSolution]], Protocol[_TSolution]): ...


class _Neighbourhood(SupportsSubNeighbourhoods[Iterable[_SubNeighbourhood[_TSolution]]], Protocol[_TSolution]): ...


class _Problem(SupportsLocalNeighbourhood[_Neighbourhood[_TSolution]], Protocol[_TSolution]): ...


def vns_fi(problem: _Problem[_TSolution], solution: _TSolution) -> _TSolution:
    local_nbhood = problem.local_neighbourhood()
    sub_nbhood = tuple(local_nbhood.sub_neighbourhoods())

    nn = len(sub_nbhood)
    j = 0
    while j < nn:
        vi = iter(sub_nbhood[j].random_moves_without_replacement(solution))
        v = next(vi, None)
        while v is not None:
            incr = v.objective_value_increment(solution)
            assert incr is not None
            if incr < 0:
                log.info(f"Increment: {incr}")
                j = 0
                solution = v.apply_move(solution)
                vi = iter(sub_nbhood[j].random_moves_without_replacement(solution))
            v = next(vi, None)
        j += 1
    return solution
