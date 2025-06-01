# SPDX-FileCopyrightText: © 2025 Authors of the roar-net-api-py project <https://github.com/roar-net/roar-net-api-py/blob/main/AUTHORS>
#
# SPDX-License-Identifier: Apache-2.0

from logging import getLogger
from time import perf_counter
from typing import Protocol, TypeVar

from ..operations import (
    SupportsApplyMove,
    SupportsLocalNeighbourhood,
    SupportsObjectiveValueIncrement,
    SupportsRandomMove,
)

log = getLogger(__name__)


_TSolution = TypeVar("_TSolution")


class _Move(SupportsApplyMove[_TSolution], SupportsObjectiveValueIncrement[_TSolution], Protocol): ...


class _Neighbourhood(SupportsRandomMove[_TSolution, _Move[_TSolution]], Protocol): ...


class _Problem(SupportsLocalNeighbourhood[_Neighbourhood[_TSolution]], Protocol): ...


def rls(problem: _Problem[_TSolution], solution: _TSolution, budget: float) -> _TSolution:
    start = perf_counter()

    neigh = problem.local_neighbourhood()

    while perf_counter() - start < budget:
        move = neigh.random_move(solution)

        if move is None:
            break

        increment = move.objective_value_increment(solution)
        assert increment is not None

        if increment < 0:
            log.info(f"Found increment: {increment}")
            solution = move.apply_move(solution)

    return solution
