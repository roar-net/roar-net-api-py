# SPDX-FileCopyrightText: © 2025 Authors of the roar-net-api-py project <https://github.com/roar-net/roar-net-api-py/blob/main/AUTHORS>
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable
from typing import Protocol, Union

from ..api.operations import (
    SupportsApplyMove,
    SupportsLocalNeighbourhood,
    SupportsObjectiveValueIncrement,
    SupportsRandomMovesWithoutReplacement,
)


class Solution(Protocol): ...


class Move(SupportsApplyMove[Solution], SupportsObjectiveValueIncrement[Solution], Protocol): ...


class Neighbourhood(SupportsRandomMovesWithoutReplacement[Solution, Move], Protocol): ...


class Problem(SupportsLocalNeighbourhood[Neighbourhood], Protocol): ...


def first_improvement(problem: Problem, solution: Solution):
    # modifies solution in place and returns a reference to it
    neigh = problem.local_neighbourhood()

    move_iter = iter(_valid_moves_and_increments(neigh, solution))
    move_and_incr = next(move_iter, None)
    while move_and_incr is not None:
        move, increment = move_and_incr

        # IMPROVE: dealing with tolerances
        if increment < 0:
            move.apply_move(solution)
            move_iter = iter(_valid_moves_and_increments(neigh, solution))

        move_and_incr = next(move_iter, None)

    return solution


def _valid_moves_and_increments(neigh: Neighbourhood, solution: Solution) -> Iterable[tuple[Move, Union[int, float]]]:
    for move in neigh.random_moves_without_replacement(solution):
        incr = move.objective_value_increment(solution)
        if incr is not None:
            yield (move, incr)
