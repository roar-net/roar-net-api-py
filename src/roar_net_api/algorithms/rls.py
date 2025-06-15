# SPDX-FileCopyrightText: © 2025 Authors of the roar-net-api-py project <https://github.com/roar-net/roar-net-api-py/blob/main/AUTHORS>
#
# SPDX-License-Identifier: Apache-2.0

from logging import getLogger
from time import perf_counter
from typing import Protocol, TypeVar

from ..values import Float
from ..operations import (
    SupportsApplyMove,
    SupportsLocalNeighbourhood,
    SupportsObjectiveValueIncrement,
    SupportsRandomMovesWithoutReplacement,
)

log = getLogger(__name__)


_Increment = int | float | Float

_TSolution = TypeVar("_TSolution")


class _Move(SupportsApplyMove[_TSolution], SupportsObjectiveValueIncrement[_TSolution, _Increment], Protocol): ...


class _Neighbourhood(SupportsRandomMovesWithoutReplacement[_TSolution, _Move[_TSolution]], Protocol): ...


class _Problem(SupportsLocalNeighbourhood[_Neighbourhood[_TSolution]], Protocol): ...


def rls(problem: _Problem[_TSolution], solution: _TSolution, budget: float) -> _TSolution:
    start = perf_counter()

    neigh = problem.local_neighbourhood()

    while perf_counter() - start < budget:
        for move in neigh.random_moves_without_replacement(solution):
            incr = move.objective_value_increment(solution)
            assert incr is not None
            if incr <= 0:
                log.info(f"Found increment: {incr}")
                solution = move.apply_move(solution)
                break
            if perf_counter() - start >= budget:
                return solution
        else:
            break

    return solution
