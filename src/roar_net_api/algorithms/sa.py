# SPDX-FileCopyrightText: © 2025 Authors of the roar-net-api-py project <https://github.com/roar-net/roar-net-api-py/blob/main/AUTHORS>
#
# SPDX-License-Identifier: Apache-2.0

import random
from math import exp
from time import perf_counter
from typing import Callable, Optional, Protocol

from ..api.operations import (
    SupportsApplyMove,
    SupportsCopySolution,
    SupportsLocalNeighbourhood,
    SupportsObjectiveValue,
    SupportsObjectiveValueIncrement,
    SupportsRandomMovesWithoutReplacement,
)


class Solution(SupportsCopySolution, SupportsObjectiveValue, Protocol): ...


class Move(SupportsApplyMove[Solution], SupportsObjectiveValueIncrement[Solution], Protocol): ...


class Neighbourhood(SupportsRandomMovesWithoutReplacement[Solution, Move], Protocol): ...


class Problem(SupportsLocalNeighbourhood[Neighbourhood], Protocol): ...


class LinearDecay:
    def __init__(self, init_temp: float) -> None:
        self.init_temp = init_temp

    def __call__(self, t: float) -> float:
        return t * self.init_temp


class ExponentialAcceptance:
    def __init__(self) -> None: ...

    def __call__(self, incr: float, t: float) -> float:
        if incr <= 0:
            return 1.0
        else:
            return exp(-incr / t)


def sa(
    problem: Problem,
    solution: Solution,
    budget: float,
    init_temp: float,
    temperature: Optional[Callable[[float], float]] = None,
    acceptance: Optional[Callable[[float, float], float]] = None,
) -> Solution:
    if temperature is None:
        temperature = LinearDecay(init_temp)

    if acceptance is None:
        acceptance = ExponentialAcceptance()

    start = perf_counter()
    neigh = problem.local_neighbourhood()
    best = solution.copy_solution()
    bestobj = best.objective_value()
    while perf_counter() - start < budget:
        for move in neigh.random_moves_without_replacement(solution):
            t = temperature(1 - (perf_counter() - start) / budget)
            if t <= 0:
                break
            incr = move.objective_value_increment(solution)
            if incr is None:
                continue
            if acceptance(incr, t) >= random.random():
                move.apply_move(solution)
                obj = solution.objective_value()
                assert obj is not None

                if bestobj is None or obj < bestobj:
                    best = solution.copy_solution()
                    bestobj = obj
                break
    return best
