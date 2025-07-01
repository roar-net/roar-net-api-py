# SPDX-FileCopyrightText: © 2025 Authors of the roar-net-api-py project <https://github.com/roar-net/roar-net-api-py/blob/main/AUTHORS>
#
# SPDX-License-Identifier: Apache-2.0

import random
from collections.abc import Callable
from logging import getLogger
from operator import itemgetter
from time import perf_counter
from typing import Optional, Protocol, TypeVar, Union, cast

from ..operations import (
    SupportsApplyMove,
    SupportsConstructionNeighbourhood,
    SupportsCopySolution,
    SupportsEmptySolution,
    SupportsLowerBoundIncrement,
    SupportsMoves,
    SupportsObjectiveValue,
)

log = getLogger(__name__)


class _Solution(SupportsObjectiveValue, SupportsCopySolution, Protocol): ...


_TSolution = TypeVar("_TSolution", bound=_Solution)


class _Move(SupportsLowerBoundIncrement[_TSolution], SupportsApplyMove[_TSolution], Protocol): ...


class _Neighbourhood(SupportsMoves[_TSolution, _Move[_TSolution]], Protocol): ...


class _Problem(
    SupportsConstructionNeighbourhood[_Neighbourhood[_TSolution]], SupportsEmptySolution[_TSolution], Protocol
): ...


LocalSearchFunc = Callable[[_Problem[_TSolution], _TSolution], _TSolution]


def grasp(
    problem: _Problem[_TSolution],
    budget: float,
    solution: Optional[_TSolution] = None,
    alpha: float = 0.1,
    local_search: Optional[LocalSearchFunc[_TSolution]] = None,
) -> _TSolution:
    start = perf_counter()

    neigh = problem.construction_neighbourhood()

    if solution is None:
        solution = problem.empty_solution()

    best = solution
    bestobj = solution.objective_value()

    while perf_counter() - start < budget:
        s = solution.copy_solution()
        b = None
        bobj = None

        cl = _valid_moves_and_increments(neigh, s)
        while len(cl) != 0:
            cmin = min(cl, key=itemgetter(1))[1]
            cmax = max(cl, key=itemgetter(1))[1]
            thresh = cmin + alpha * (cmax - cmin)
            rcl = [m for m, decr in cl if decr <= thresh]
            m = random.choice(rcl)
            s = m.apply_move(s)
            obj = s.objective_value()
            if obj is not None and (bobj is None or obj < bobj):
                b = s.copy_solution()
                bobj = b.objective_value()
            cl = _valid_moves_and_increments(neigh, s)
        if b is not None:
            if local_search is not None:
                b = local_search(problem, b)
                # The following assumes that local_search returns a better or equal objective value
                bobj = cast(Union[int, float], b.objective_value())
            bobj = cast(Union[int, float], bobj)
            if bestobj is None or bobj < bestobj:
                log.info(f"Best solution: {bobj}")
                best = b
                bestobj = bobj
    return best


def _valid_moves_and_increments(
    neigh: _Neighbourhood[_TSolution], solution: _TSolution
) -> list[tuple[_Move[_TSolution], Union[int, float]]]:
    res: list[tuple[_Move[_TSolution], Union[int, float]]] = []
    for m in neigh.moves(solution):
        incr = m.lower_bound_increment(solution)
        if incr is not None:
            res.append((m, incr))
    return res
