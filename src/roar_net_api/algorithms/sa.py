# SPDX-FileCopyrightText: © 2025 Authors of the roar-net-api-py project <https://github.com/roar-net/roar-net-api-py/blob/main/AUTHORS>
#
# SPDX-License-Identifier: Apache-2.0

import random
from logging import getLogger
from math import exp
from time import perf_counter
from typing import Callable, Optional, Protocol, TypeVar

from ..values import Float
from ..operations import (
    SupportsApplyMove,
    SupportsCopySolution,
    SupportsLocalNeighbourhood,
    SupportsObjectiveValue,
    SupportsObjectiveValueIncrement,
    SupportsRandomMovesWithoutReplacement,
)

log = getLogger(__name__)

_Value = int | float | Float

_Increment = int | float | Float


class _Solution(SupportsCopySolution, SupportsObjectiveValue[_Value], Protocol): ...


_TSolution = TypeVar("_TSolution", bound=_Solution)


class _Move(SupportsApplyMove[_TSolution], SupportsObjectiveValueIncrement[_TSolution, _Increment], Protocol): ...


class _Neighbourhood(SupportsRandomMovesWithoutReplacement[_TSolution, _Move[_TSolution]], Protocol): ...


class _Problem(SupportsLocalNeighbourhood[_Neighbourhood[_TSolution]], Protocol): ...


class LinearDecay:
    def __init__(self, init_temp: float) -> None:
        self.init_temp = init_temp

    def __call__(self, t: float) -> float:
        return t * self.init_temp


class ExponentialAcceptance:
    def __init__(self) -> None: ...

    def __call__(self, incr: _Value, t: float) -> float:
        if incr <= 0:
            return 1.0
        else:
            return exp(-float(incr) / t)


def sa(
    problem: _Problem[_TSolution],
    solution: _TSolution,
    budget: float,
    init_temp: float,
    temperature: Optional[Callable[[float], float]] = None,
    acceptance: Optional[Callable[[_Value, float], float]] = None,
) -> _TSolution:
    if temperature is None:
        temperature = LinearDecay(init_temp)

    if acceptance is None:
        acceptance = ExponentialAcceptance()

    start = perf_counter()

    neigh = problem.local_neighbourhood()

    best = solution.copy_solution()
    bestobj = best.objective_value()
    assert bestobj is not None

    while perf_counter() - start < budget:
        for move in neigh.random_moves_without_replacement(solution):
            t = temperature(1 - (perf_counter() - start) / budget)
            if t <= 0:
                break
            incr = move.objective_value_increment(solution)
            assert incr is not None

            if acceptance(incr, t) >= random.random():
                move.apply_move(solution)
                obj = solution.objective_value()
                assert obj is not None

                if obj < float(bestobj):
                    log.info(f"Best solution: {obj}")
                    best = solution.copy_solution()
                    bestobj = obj
                break
    return best
