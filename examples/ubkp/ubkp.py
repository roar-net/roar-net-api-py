#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: © 2025 Authors of the roar-net-api-py project <https://github.com/roar-net/roar-net-api-py/blob/main/AUTHORS>
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
import logging
import random
import sys
from collections.abc import Iterable, Sequence
from logging import getLogger
from typing import Optional, Protocol, Self, TextIO, TypeVar, final

from roar_net_api.values import Float, Pareto
from roar_net_api.operations import (
    SupportsApplyMove,
    SupportsCopySolution,
    SupportsEmptySolution,
    SupportsLocalNeighbourhood,
    SupportsMoves,
    SupportsObjectiveValue,
    SupportsObjectiveValueIncrement,
    SupportsRandomMove,
    SupportsRandomMovesWithoutReplacement,
)

log = getLogger(__name__)


class _SupportsLT(Protocol):
    def __lt__(self, other: Self) -> bool: ...


_T = TypeVar("_T", bound=_SupportsLT)


def argmin(seq: Sequence[_T]) -> int:
    return min(range(len(seq)), key=seq.__getitem__)


def sparse_fisher_yates_iter(n: int) -> Iterable[int]:
    p: dict[int, int] = dict()
    for i in range(n - 1, -1, -1):
        r = random.randrange(i + 1)
        yield p.get(r, r)
        if i != r:
            # p[r] = p.pop(i, i) # saves memory, takes time
            p[r] = p.get(i, i)  # lazy, but faster


# --- Custom Float constructor ---


def myfloat(f: float) -> Float:
    return Float(f, rtol=0.0, atol=1e-6)


# --- Knapsack item ---


@dataclass
class Item:
    ord: int  # order
    value: float
    weight: float


# ---------------------------------- Solution --------------------------------


@final
class Solution(SupportsCopySolution, SupportsObjectiveValue[Pareto[Float]]):
    def __init__(
        self,
        problem: Problem,
        used: list[Item],
        unused: list[Item],
        value: Optional[float] = None,
        weight: Optional[float] = None,
    ):
        self.problem = problem
        self.used = used
        self.unused = unused
        self.value = value if value is not None else sum(map(lambda i: i.value, self.used))
        self.weight = weight if weight is not None else sum(map(lambda i: i.weight, self.used))

    def __str__(self) -> str:
        return " ".join(map(lambda i: str(i.ord + 1), sorted(self.used, key=lambda item: item.ord)))

    def to_textio(self, f: TextIO) -> None:
        f.write(str(self) + "\n")

    def copy_solution(self) -> Self:
        return type(self)(
            problem=self.problem,
            used=self.used.copy(),
            unused=self.unused.copy(),
            value=self.value,
            weight=self.weight,
        )

    def objective_value(self) -> Pareto[Float]:
        return Pareto((myfloat(-self.value), myfloat(self.weight)))

    def add_item(self, ix: int) -> None:
        item = self.unused[ix]

        self.unused[ix] = self.unused[-1]
        self.unused.pop()
        self.used.append(item)

        self.value += item.value
        self.weight += item.weight

    def rem_item(self, ix: int) -> None:
        item = self.used[ix]

        self.used[ix] = self.used[-1]
        self.used.pop()
        self.unused.append(item)

        self.value -= item.value
        self.weight -= item.weight


# ----------------------------------- Moves -----------------------------------


@final
class AddMove(SupportsApplyMove[Solution], SupportsObjectiveValueIncrement[Solution, Pareto[Float]]):
    """
    Move to add a new item to the knapsack.
    """

    def __init__(self, ix: int):
        self.ix = ix

    def apply_move(self, solution: Solution) -> Solution:
        solution.add_item(self.ix)
        return solution

    def objective_value_increment(self, solution: Solution) -> Pareto[Float]:
        item = solution.unused[self.ix]
        return Pareto((myfloat(-item.value), myfloat(item.weight)))


@final
class SwapMove(SupportsApplyMove[Solution], SupportsObjectiveValueIncrement[Solution, Pareto[Float]]):
    """
    Move to swap items in the knapsack. Removes `i` and inserts `j`
    """

    def __init__(self, ix: int, jx: int):
        self.ix = ix
        self.jx = jx

    def apply_move(self, solution: Solution) -> Solution:
        solution.rem_item(self.ix)
        solution.add_item(self.jx)
        return solution

    def objective_value_increment(self, solution: Solution) -> Pareto[Float]:
        i = solution.used[self.ix]
        iv = i.value
        iw = i.weight
        j = solution.unused[self.jx]
        jv = j.value
        jw = j.weight
        return Pareto((myfloat(iv - jv), myfloat(jw - iw)))


# ------------------------------- Neighbourhood ------------------------------


@final
class LocalNeighbourhood(
    SupportsMoves[Solution, AddMove | SwapMove],
    SupportsRandomMovesWithoutReplacement[Solution, AddMove | SwapMove],
    SupportsRandomMove[Solution, AddMove | SwapMove],
):
    def moves(self, solution: Solution) -> Iterable[AddMove | SwapMove]:
        for ix, _ in enumerate(solution.unused):
            yield AddMove(ix)
        for ix, _ in enumerate(solution.used):
            for jx, _ in enumerate(solution.unused):
                yield SwapMove(ix, jx)

    def random_moves_without_replacement(self, solution: Solution) -> Iterable[AddMove | SwapMove]:
        l1 = len(solution.used)
        l2 = len(solution.unused)
        nadd = l2
        nswap = l1 * l2
        for k in sparse_fisher_yates_iter(nswap + nadd):
            if k < nswap:
                ix, jx = divmod(k, l2)
                yield SwapMove(ix, jx)
            else:
                ix = k - nswap
                yield AddMove(ix)

    def random_move(self, solution: Solution) -> Optional[AddMove | SwapMove]:
        return next(iter(self.random_moves_without_replacement(solution)), None)


# ---------------------------------- Problem --------------------------------


@final
class Problem(
    SupportsLocalNeighbourhood[LocalNeighbourhood],
    SupportsEmptySolution[Solution],
):
    def __init__(self, items: tuple[Item, ...]):
        self.items = items
        self.l_nbhood: Optional[LocalNeighbourhood] = None

    def local_neighbourhood(self) -> LocalNeighbourhood:
        if self.l_nbhood is None:
            self.l_nbhood = LocalNeighbourhood()
        return self.l_nbhood

    @classmethod
    def from_textio(cls, f: TextIO) -> Self:
        """
        Create a problem from a text I/O source `f` in TSPLIB format
        """
        n, o, c = map(int, f.readline().strip().split())
        assert o == 1
        assert c == 1
        items: list[Item] = []
        for i in range(n):
            v, w = map(float, f.readline().strip().split())
            items.append(Item(ord=i, value=v, weight=w))
        return cls(tuple(items))

    def empty_solution(self) -> Solution:
        return Solution(self, [], list(self.items))


if __name__ == "__main__":
    import roar_net_api.algorithms as alg

    logging.basicConfig(stream=sys.stderr, level="INFO", format="%(levelname)s;%(asctime)s;%(message)s")

    problem = Problem.from_textio(sys.stdin)

    # Run pareto local search
    archive = alg.pls(problem, problem.empty_solution(), 30.0)
    archive.sort(key=lambda s: s.objective_value()[0])

    log.info("Objective values after pareto local search")
    for solution in archive:
        log.info(str(solution.objective_value()))

    # Print the final solution to stdout
    print(len(archive))
    for solution in archive:
        solution.to_textio(sys.stdout)
