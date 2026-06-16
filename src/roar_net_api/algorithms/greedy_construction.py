# SPDX-FileCopyrightText: © 2025 Authors of the roar-net-api-py project <https://github.com/roar-net/roar-net-api-py/blob/main/AUTHORS>
#
# SPDX-License-Identifier: Apache-2.0

import math
import random
from collections.abc import Iterable
from logging import getLogger
from typing import Optional, Protocol, TypeVar, Union

from ..operations import (
    SupportsApplyMove,
    SupportsConstructionNeighbourhood,
    SupportsEmptySolution,
    SupportsLowerBoundIncrement,
    SupportsMoves,
)

log = getLogger(__name__)


_TSolution = TypeVar("_TSolution")


class _Move(SupportsApplyMove[_TSolution], SupportsLowerBoundIncrement[_TSolution], Protocol): ...


class _Neighbourhood(SupportsMoves[_TSolution, _Move[_TSolution]], Protocol): ...


class _Problem(
    SupportsEmptySolution[_TSolution], SupportsConstructionNeighbourhood[_Neighbourhood[_TSolution]], Protocol
): ...


def greedy_construction(problem: _Problem[_TSolution], solution: Optional[_TSolution] = None) -> _TSolution:
    """
    Solves `problem` using a greedy construction approach.

    Note: if `solution` is given it must be a solution to the given `problem` object.
    """
    neigh = problem.construction_neighbourhood()

    if solution is None:
        solution = problem.empty_solution()

    # Utility functions for reservoir sampling (Algorithm L)
    def gen_w() -> float:
        return math.exp(math.log(random.random()))

    def gen_j(w: float) -> int:
        return math.floor(math.log(random.random()) / math.log(1 - w)) + 1

    move_iter = iter(_valid_moves_and_increments(neigh, solution))
    move_and_incr = next(move_iter, None)
    while move_and_incr is not None:
        best_move, best_incr = move_and_incr

        rs_i = 0
        rs_w = 1.0
        rs_j = 0

        for move, incr in move_iter:
            if incr < best_incr:
                best_move = move
                best_incr = incr
                rs_i = 0
                rs_w = 1
                rs_j = 0
            elif incr == best_incr:
                rs_i += 1
                if rs_i > rs_j:
                    rs_w *= gen_w()
                    rs_j += gen_j(rs_w)
                if rs_i == rs_j:
                    best_move = move
                    best_incr = incr

        solution = best_move.apply_move(solution)

        move_iter = iter(_valid_moves_and_increments(neigh, solution))
        move_and_incr = next(move_iter, None)

    return solution


def _valid_moves_and_increments(
    neigh: _Neighbourhood[_TSolution], solution: _TSolution
) -> Iterable[tuple[_Move[_TSolution], Union[int, float]]]:
    for move in neigh.moves(solution):
        incr = move.lower_bound_increment(solution)
        if incr is not None:
            yield (move, incr)
