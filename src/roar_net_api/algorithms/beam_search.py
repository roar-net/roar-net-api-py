# SPDX-FileCopyrightText: © 2025 Authors of the roar-net-api-py project <https://github.com/roar-net/roar-net-api-py/blob/main/AUTHORS>
#
# SPDX-License-Identifier: Apache-2.0

import bisect
from collections.abc import Callable
from logging import getLogger
from operator import itemgetter
from typing import Generic, Optional, Protocol, Self, TypeVar, Union, cast

from ..operations import (
    SupportsApplyMove,
    SupportsConstructionNeighbourhood,
    SupportsCopySolution,
    SupportsEmptySolution,
    SupportsLowerBound,
    SupportsLowerBoundIncrement,
    SupportsMoves,
    SupportsObjectiveValue,
)

log = getLogger(__name__)


class _Solution(SupportsLowerBound, SupportsObjectiveValue, SupportsCopySolution, Protocol): ...


_TSolution = TypeVar("_TSolution", bound=_Solution)


class _Move(SupportsLowerBoundIncrement[_TSolution], SupportsApplyMove[_TSolution], Protocol): ...


class _Neighbourhood(SupportsMoves[_TSolution, _Move[_TSolution]], Protocol): ...


class _Problem(
    SupportsConstructionNeighbourhood[_Neighbourhood[_TSolution]], SupportsEmptySolution[_TSolution], Protocol
): ...


BSList = list[tuple[Union[int, float], _TSolution]]


class KeyProtocol(Protocol):
    def __lt__(self, value: Self, /) -> bool: ...


Key = TypeVar("Key", bound=KeyProtocol)
Value = TypeVar("Value")
KeyFunc = Callable[[Value], Key]


class KMin(Generic[Key, Value]):
    """
    Class to keep a set of the k min objects according to a key
    function
    """

    def __init__(self, k: int, key: KeyFunc[Value, Key]):
        self.k = k
        self.key = key
        self.keys: list[Key] = []
        self.values: list[Value] = []

    def insert(self, value: Value):
        key = self.key(value)
        if len(self.values) == self.k:
            if key > self.keys[-1]:
                return
        i = bisect.bisect_right(self.keys, key)
        self.keys.insert(i, key)
        self.values.insert(i, value)
        if len(self.values) > self.k:
            self.keys.pop()
            self.values.pop()

    def __iter__(self):
        return self.values.__iter__()

    def __len__(self):
        return self.values.__len__()


def beam_search(problem: _Problem[_TSolution], solution: Optional[_TSolution] = None, bw: int = 10) -> _TSolution:
    neigh = problem.construction_neighbourhood()

    if solution is None:
        solution = problem.empty_solution()

    best = solution
    bestobj = best.objective_value()

    lb = best.lower_bound()

    if lb is None:
        return best

    v: BSList[_TSolution] = [(lb, best)]

    while True:
        candidates = KMin[Union[int, float], tuple[Union[int, float], _TSolution, _Move[_TSolution]]](
            bw, key=itemgetter(0)
        )
        for lb, s in v:
            for m in neigh.moves(s):
                incr = m.lower_bound_increment(s)
                if incr is not None:
                    candidates.insert((lb + incr, s, m))

        if len(candidates) == 0:
            break

        v = []
        for lb, s, m in candidates:
            ns = s.copy_solution()
            ns = m.apply_move(ns)
            v.append((cast(Union[int, float], lb), ns))
            obj = ns.objective_value()
            if obj is not None and (bestobj is None or obj < bestobj):
                log.info(f"Best solution: {obj}")
                best = ns
                bestobj = obj

    return best
