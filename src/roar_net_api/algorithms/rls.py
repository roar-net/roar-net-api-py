# SPDX-FileCopyrightText: © 2025 Authors of the roar-net-api-py project <https://github.com/roar-net/roar-net-api-py/blob/main/AUTHORS>
#
# SPDX-License-Identifier: Apache-2.0

from logging import getLogger
from time import perf_counter
from typing import Protocol, TypeVar

from ..operations import (
    SupportsApplyMove,
    SupportsLocalNeighbourhood,
    SupportsObjectiveValue,
    SupportsObjectiveValueIncrement,
    SupportsRandomMovesWithoutReplacement,
)
from ..utils.logging import PerformanceLogger

log = getLogger(__name__)

perflogger = PerformanceLogger()


class _Solution(SupportsObjectiveValue, Protocol): ...


_TSolution = TypeVar("_TSolution", bound=_Solution)


class _Move(SupportsApplyMove[_TSolution], SupportsObjectiveValueIncrement[_TSolution], Protocol): ...


class _Neighbourhood(SupportsRandomMovesWithoutReplacement[_TSolution, _Move[_TSolution]], Protocol): ...


class _Problem(SupportsLocalNeighbourhood[_Neighbourhood[_TSolution]], Protocol): ...


def rls(
    problem: _Problem[_TSolution],
    solution: _TSolution,
    budget: float,
) -> _TSolution:
    start = perf_counter()

    neigh = problem.local_neighbourhood()

    if perflogger.active:
        perflogger.log(solution.objective_value(), __name__)

    while perf_counter() - start < budget:
        for move in neigh.random_moves_without_replacement(solution):
            incr = move.objective_value_increment(solution)
            assert incr is not None
            if incr <= 0:
                log.info(f"Found increment: {incr}")
                solution = move.apply_move(solution)
                if perflogger.active:
                    perflogger.log(solution.objective_value(), __name__)
                break
            if perf_counter() - start >= budget:
                return solution
        else:
            break

    return solution
