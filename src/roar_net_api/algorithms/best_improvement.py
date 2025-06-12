# SPDX-FileCopyrightText: © 2025 Authors of the roar-net-api-py project <https://github.com/roar-net/roar-net-api-py/blob/main/AUTHORS>
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable
from logging import getLogger
from typing import Protocol, TypeVar, Union

from ..operations import (
    SupportsApplyMove,
    SupportsLocalNeighbourhood,
    SupportsMoves,
    SupportsObjectiveValueIncrement,
)

log = getLogger(__name__)


_TSolution = TypeVar("_TSolution")


class _Move(SupportsApplyMove[_TSolution], SupportsObjectiveValueIncrement[_TSolution], Protocol): ...


class _Neighbourhood(SupportsMoves[_TSolution, _Move[_TSolution]], Protocol): ...


class _Problem(SupportsLocalNeighbourhood[_Neighbourhood[_TSolution]], Protocol): ...


def best_improvement(problem: _Problem[_TSolution], solution: _TSolution) -> _TSolution:
    neigh = problem.local_neighbourhood()

    move_iter = iter(_valid_moves_and_increments(neigh, solution))
    move_and_incr = next(move_iter, None)
    while move_and_incr is not None:
        best_move, best_incr = move_and_incr

        for move, incr in move_iter:
            if incr < best_incr:
                best_move = move
                best_incr = incr

        log.info(f"Best increment: {best_incr}")

        best_move.apply_move(solution)

        move_iter = iter(_valid_moves_and_increments(neigh, solution))
        move_and_incr = next(move_iter, None)

    return solution


def _valid_moves_and_increments(
    neigh: _Neighbourhood[_TSolution], solution: _TSolution
) -> Iterable[tuple[_Move[_TSolution], Union[int, float]]]:
    for move in neigh.moves(solution):
        incr = move.objective_value_increment(solution)
        assert incr is not None
        if incr < 0:
            yield (move, incr)
