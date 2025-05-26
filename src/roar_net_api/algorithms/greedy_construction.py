# SPDX-FileCopyrightText: © 2025 Authors of the roar-net-api-py project <https://github.com/roar-net/roar-net-api-py/blob/main/AUTHORS>
#
# SPDX-License-Identifier: Apache-2.0

import random
from collections.abc import Iterable
from typing import Optional, Protocol, Union

from ..api.operations import (
    SupportsApplyMove,
    SupportsConstructionNeighbourhood,
    SupportsEmptySolution,
    SupportsLowerBoundIncrement,
    SupportsMoves,
)


class Solution(Protocol): ...


class Move(SupportsLowerBoundIncrement[Solution], SupportsApplyMove[Solution], Protocol): ...


class Neighbourhood(SupportsMoves[Solution, Move], Protocol): ...


class Problem(SupportsConstructionNeighbourhood[Neighbourhood], SupportsEmptySolution[Solution], Protocol): ...


def greedy_construction(problem: Problem, solution: Optional[Solution] = None) -> Solution:
    """
    Solves `problem` using a greedy construction approach.

    Note: if `solution` is given it must be a solution to `problem`. Otherwise, an empty solution is generated.
    """
    neigh = problem.construction_neighbourhood()

    if solution is None:
        solution = problem.empty_solution()

    move_iter = iter(_valid_moves_and_increments(neigh, solution))
    move_and_incr = next(move_iter, None)
    while move_and_incr is not None:
        best_move, best_incr = move_and_incr

        for move, incr in move_iter:
            if incr < best_incr:
                best_move = move
                best_incr = incr
                # IMPROVE: if incr is a float, how to deal with values close to zero
                # Should we have a tolerance parameter?
                if incr == 0:
                    break

        best_move.apply_move(solution)

        move_iter = iter(_valid_moves_and_increments(neigh, solution))
        move_and_incr = next(move_iter, None)

    return solution


# IMPROVE: this reuses a lot of the code from the above. Maybe we should make random tie breaking a parameter?
def greedy_construction_with_random_tie_breaking(problem: Problem, solution: Optional[Solution] = None) -> Solution:
    """
    Solves `problem` using a greedy construction approach.

    Note: if `solution` is given it must be a solution to `problem`. Otherwise, an empty solution is generated.
    """
    neigh = problem.construction_neighbourhood()

    if solution is None:
        solution = problem.empty_solution()

    move_iter = iter(_valid_moves_and_increments(neigh, solution))
    move_and_incr = next(move_iter, None)
    while move_and_incr is not None:
        best_moves = [move_and_incr[0]]
        best_incr = move_and_incr[1]

        for move, incr in move_iter:
            if incr < best_incr:
                best_moves = [move]
                best_incr = incr
            # IMPROVE: How should we deal with tolerance? Do we want this to be handled by the
            # Value object in the future?
            elif incr < best_incr + 1e-6:
                best_moves.append(move)

        # IMPROVE: Make random selection function configurable
        random.choice(best_moves).apply_move(solution)

        move_iter = iter(_valid_moves_and_increments(neigh, solution))
        move_and_incr = next(move_iter, None)

    return solution


def _valid_moves_and_increments(neigh: Neighbourhood, solution: Solution) -> Iterable[tuple[Move, Union[int, float]]]:
    for move in neigh.moves(solution):
        incr = move.lower_bound_increment(solution)
        if incr is not None:
            yield (move, incr)
