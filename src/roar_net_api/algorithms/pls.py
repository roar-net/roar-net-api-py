# SPDX-FileCopyrightText: © 2025 Authors of the roar-net-api-py project <https://github.com/roar-net/roar-net-api-py/blob/main/AUTHORS>
#
# SPDX-License-Identifier: Apache-2.0

from itertools import chain
from logging import getLogger
from operator import itemgetter
from time import perf_counter
from typing import Protocol, TypeVar
import random

from ..values import Pareto, Float
from ..operations import (
    SupportsLocalNeighbourhood,
    SupportsObjectiveValue,
    SupportsMoves,
    SupportsApplyMove,
    SupportsObjectiveValueIncrement,
    SupportsCopySolution,
)

log = getLogger(__name__)

_Value = Pareto[int] | Pareto[float] | Pareto[Float]

_Increment = Pareto[int] | Pareto[float] | Pareto[Float]


class _Solution(SupportsObjectiveValue[_Value], SupportsCopySolution, Protocol): ...


_TSolution = TypeVar("_TSolution", bound=_Solution)


class _Move(
    SupportsApplyMove[_TSolution], SupportsObjectiveValueIncrement[_TSolution, _Increment], Protocol[_TSolution]
): ...


class _LocalNeighbourhood(SupportsMoves[_TSolution, _Move[_TSolution]], Protocol[_TSolution]): ...


class _Problem(SupportsLocalNeighbourhood[_LocalNeighbourhood[_TSolution]], Protocol[_TSolution]): ...


_IList = list[tuple[_Value, _TSolution]]


_T = TypeVar("_T")


def swap_pop(v: list[_T], indx: int) -> _T:
    if indx < len(v) - 1:
        v[indx], v[-1] = v[-1], v[indx]
    return v.pop()


def pls(problem: _Problem[_TSolution], solution: _TSolution, budget: float) -> list[_TSolution]:
    # Returns a list of non-dominated (value, solution) pairs
    # FIXME: Lists (archive and new) should probably be modified in place
    start = perf_counter()

    local_nbhood = problem.local_neighbourhood()

    archive: _IList[_TSolution] = []

    obj = solution.objective_value()
    assert obj is not None

    new: _IList[_TSolution] = [(obj, solution)]

    count = 0
    while len(new) and perf_counter() - start < budget:
        count += 1
        val, sol = swap_pop(new, random.randrange(len(new)))
        archive.append((val, sol))
        for move in iter(local_nbhood.moves(sol)):
            incr = move.objective_value_increment(sol)
            assert incr is not None
            new_val = val + incr  # type: ignore # these will be the same type
            if val <= new_val:  # type: ignore # these will be the same type
                # Neighbour is weakly dominated by current solution
                continue
            # Check archives
            accept = True
            for lst in (archive, new):
                new_lst: list[tuple[_Value, _TSolution]] = []
                for i in range(len(lst)):
                    val_i, sol_i = lst[i]
                    if val_i <= new_val:  # type: ignore # these will be the same type
                        accept = False
                        new_lst.extend(lst[i:])
                        break
                    if new_val < val_i:  # type: ignore # these will be the same type
                        continue
                    new_lst.append((val_i, sol_i))
                lst[:] = new_lst
                if not accept:
                    break
            # Insert if non-dominated
            if accept:
                cpy = move.apply_move(sol.copy_solution())
                obj = cpy.objective_value()
                assert obj is not None
                new.append((obj, cpy))
                log.debug(f"Accepted solution with objective value: {obj}")
        log.info(f"Iteration {count}, archive size: {len(archive) + len(new)}")
    return list(map(itemgetter(1), chain(archive, new)))
