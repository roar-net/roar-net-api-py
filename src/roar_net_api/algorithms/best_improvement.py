# SPDX-FileCopyrightText: © 2025 Authors of the roar-net-api-py project <https://github.com/roar-net/roar-net-api-py/blob/main/AUTHORS>
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable
from typing import Protocol, Union

from ..api.operations import (
    SupportsApplyMove,
    SupportsLocalNeighbourhood,
    SupportsMoves,
    SupportsObjectiveValueIncrement,
)


class Solution(Protocol): ...


class Move(SupportsApplyMove[Solution], SupportsObjectiveValueIncrement[Solution], Protocol): ...


class Neighbourhood(SupportsMoves[Solution, Move], Protocol): ...


class Problem(SupportsLocalNeighbourhood[Neighbourhood], Protocol): ...


def best_improvement(problem: Problem, solution: Solution) -> Solution:
    neigh = problem.local_neighbourhood()

    move_iter = iter(_valid_moves_and_increments(neigh, solution))
    move_and_incr = next(move_iter, None)
    while move_and_incr is not None:
        best_move, best_incr = move_and_incr

        for move, incr in move_iter:
            # IMPROVE: Dealing with tolerances
            if incr < best_incr:
                best_move = move
                best_incr = incr

        best_move.apply_move(solution)

        move_iter = iter(_valid_moves_and_increments(neigh, solution))
        move_and_incr = next(move_iter, None)

    return solution


def _valid_moves_and_increments(neigh: Neighbourhood, solution: Solution) -> Iterable[tuple[Move, Union[int, float]]]:
    for move in neigh.moves(solution):
        incr = move.objective_value_increment(solution)
        if incr is not None:
            yield (move, incr)
