# SPDX-FileCopyrightText: © 2025 Authors of the roar-net-api-py project <https://github.com/roar-net/roar-net-api-py/blob/main/AUTHORS>
#
# SPDX-License-Identifier: Apache-2.0

import bisect
from collections.abc import Callable
from operator import itemgetter
from typing import Generic, Optional, Protocol, Self, TypeVar, Union, cast

from ..api.operations import (
    SupportsApplyMove,
    SupportsConstructionNeighbourhood,
    SupportsCopySolution,
    SupportsEmptySolution,
    SupportsLowerBound,
    SupportsLowerBoundIncrement,
    SupportsMoves,
    SupportsObjectiveValue,
)


class Solution(SupportsLowerBound, SupportsObjectiveValue, SupportsCopySolution, Protocol): ...


class Move(SupportsLowerBoundIncrement[Solution], SupportsApplyMove[Solution], Protocol): ...


class Neighbourhood(SupportsMoves[Solution, Move], Protocol): ...


class Problem(SupportsConstructionNeighbourhood[Neighbourhood], SupportsEmptySolution[Solution], Protocol): ...


BSList = list[tuple[Union[int, float], Solution]]


class KeyProtocol(Protocol):
    def __lt__(self, value: Self, /) -> bool: ...


Key = TypeVar("Key", bound="KeyProtocol")
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
        if len(self.values) == self.k:
            key = self.key(value)
            if key < self.keys[-1]:
                i = bisect.bisect_right(self.keys, key)
                self.keys.insert(i, key)
                self.keys.pop()
                self.values.insert(i, value)
                self.values.pop()
        else:
            self.keys.append(self.key(value))
            self.values.append(value)

    def __iter__(self):
        return self.values.__iter__()

    def __len__(self):
        return self.values.__len__()


def beam_search(problem: Problem, solution: Optional[Solution] = None, bw: int = 10) -> Solution:
    neigh = problem.construction_neighbourhood()

    if solution is None:
        solution = problem.empty_solution()

    best = solution
    bestobj = best.objective_value()

    lb = best.lower_bound()

    if lb is None:
        return best

    v: BSList = [(lb, best)]

    while True:
        candidates = KMin[Union[int, float], tuple[Union[int, float], Solution, Move]](bw, key=itemgetter(0))
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
            m.apply_move(s)
            v.append((cast(Union[int, float], lb), ns))
            obj = ns.objective_value()
            if obj is not None and (bestobj is None or obj < bestobj):
                best = ns
                bestobj = obj

    return best
