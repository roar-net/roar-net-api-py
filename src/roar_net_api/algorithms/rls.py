# SPDX-FileCopyrightText: © 2025 Authors of the roar-net-api-py project <https://github.com/roar-net/roar-net-api-py/blob/main/AUTHORS>
#
# SPDX-License-Identifier: Apache-2.0

from time import perf_counter
from typing import Protocol

from ..api.operations import (
    SupportsApplyMove,
    SupportsLocalNeighbourhood,
    SupportsObjectiveValueIncrement,
    SupportsRandomMove,
)


class Solution(Protocol): ...


class Move(SupportsApplyMove[Solution], SupportsObjectiveValueIncrement[Solution], Protocol): ...


class Neighbourhood(SupportsRandomMove[Solution, Move], Protocol): ...


class Problem(SupportsLocalNeighbourhood[Neighbourhood], Protocol): ...


def rls(problem: Problem, solution: Solution, budget: float) -> Solution:
    start = perf_counter()

    neigh = problem.local_neighbourhood()

    while perf_counter() - start < budget:
        move = neigh.random_move(solution)

        if move is None:
            break

        increment = move.objective_value_increment(solution)
        assert increment is not None

        if increment < 0:
            solution = move.apply_move(solution)

    return solution
