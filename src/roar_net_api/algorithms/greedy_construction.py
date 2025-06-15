# SPDX-FileCopyrightText: © 2025 Authors of the roar-net-api-py project <https://github.com/roar-net/roar-net-api-py/blob/main/AUTHORS>
#
# SPDX-License-Identifier: Apache-2.0

import random
from collections.abc import Iterable
from logging import getLogger
from typing import Optional, Protocol, TypeVar

from ..values import Float
from ..operations import (
    SupportsApplyMove,
    SupportsConstructionNeighbourhood,
    SupportsEmptySolution,
    SupportsLowerBoundIncrement,
    SupportsMoves,
)

log = getLogger(__name__)

_Increment = int | float | Float

_TSolution = TypeVar("_TSolution")


class _Move(SupportsApplyMove[_TSolution], SupportsLowerBoundIncrement[_TSolution, _Increment], Protocol): ...


class _Neighbourhood(SupportsMoves[_TSolution, _Move[_TSolution]], Protocol): ...


class _Problem(
    SupportsEmptySolution[_TSolution], SupportsConstructionNeighbourhood[_Neighbourhood[_TSolution]], Protocol
): ...


def greedy_construction(problem: _Problem[_TSolution], solution: Optional[_TSolution] = None) -> _TSolution:
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
            if incr < best_incr:  # type: ignore
                best_move = move
                best_incr = incr
                if incr == 0:
                    break

        best_move.apply_move(solution)

        move_iter = iter(_valid_moves_and_increments(neigh, solution))
        move_and_incr = next(move_iter, None)

    return solution


# IMPROVE: this reuses a lot of the code from the above. Maybe we should make random tie breaking a parameter?
def greedy_construction_with_random_tie_breaking(
    problem: _Problem[_TSolution], solution: Optional[_TSolution] = None
) -> _TSolution:
    neigh = problem.construction_neighbourhood()

    if solution is None:
        solution = problem.empty_solution()

    move_iter = iter(_valid_moves_and_increments(neigh, solution))
    move_and_incr = next(move_iter, None)
    while move_and_incr is not None:
        best_moves = [move_and_incr[0]]
        best_incr = move_and_incr[1]

        for move, incr in move_iter:
            if incr < best_incr:  # type: ignore # incr and best_incr should be the same type
                best_moves = [move]
                best_incr = incr
            elif incr == best_incr:
                best_moves.append(move)

        log.info(f"Best increment: {best_incr}")
        random.choice(best_moves).apply_move(solution)

        move_iter = iter(_valid_moves_and_increments(neigh, solution))
        move_and_incr = next(move_iter, None)

    return solution


def _valid_moves_and_increments(
    neigh: _Neighbourhood[_TSolution], solution: _TSolution
) -> Iterable[tuple[_Move[_TSolution], _Increment]]:
    for move in neigh.moves(solution):
        incr = move.lower_bound_increment(solution)
        if incr is not None:
            yield (move, incr)
