#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: © 2025 Authors of the roar-net-api-py project <https://github.com/roar-net/roar-net-api-py/blob/main/AUTHORS>
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import random
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from logging import getLogger
from typing import Optional, Protocol, Self, TextIO, TypeVar, final

from roar_net_api.operations import (
    SupportsApplyMove,
    SupportsConstructionNeighbourhood,
    SupportsCopySolution,
    SupportsEmptySolution,
    SupportsLocalNeighbourhood,
    SupportsLowerBound,
    SupportsLowerBoundIncrement,
    SupportsMoves,
    SupportsObjectiveValue,
    SupportsObjectiveValueIncrement,
    SupportsRandomMove,
    SupportsRandomMovesWithoutReplacement,
    SupportsSubNeighbourhoods,
)
from roar_net_api.values import Float

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


# --- Lower bound ---


class LowerBound:
    def __init__(self, solution: Solution, lb: float, value: float, cap: float, idx: int):
        self.solution = solution
        self.lb = lb
        self.value = value
        self.cap = cap
        self.idx = idx

    @classmethod
    def for_solution(cls, solution: Solution) -> Self:
        v = solution.value
        c = solution.cap
        idx = 0
        lb = v
        for item in solution.problem.sorted_items:
            if item.ord in solution.used_set:
                idx += 1
                continue

            if myfloat(item.weight) <= c:
                v += item.value
                c -= item.weight
                idx += 1
                lb = v
            else:
                ratio = max(0.0, c / item.weight)
                lb += ratio * item.value
                break
        return cls(solution, lb, v, c, idx)

    def copy_to_solution(self, solution: Solution) -> LowerBound:
        return LowerBound(
            solution=solution,
            lb=self.lb,
            value=self.value,
            cap=self.cap,
            idx=self.idx,
        )

    def add_item(self, item: Item) -> None:
        item_idx = self.solution.problem.sorted_items_idx[item.ord]
        if item_idx < self.idx:
            return
        # Go backwards
        self.cap -= item.weight
        self.value += item.value
        i = self.solution.problem.sorted_items[self.idx]
        while myfloat(self.cap) < 0:
            self.idx -= 1
            assert self.idx >= 0
            i = self.solution.problem.sorted_items[self.idx]
            if i.ord != item.ord and i.ord not in self.solution.used_set:
                self.cap += i.weight
                self.value -= i.value
        # idx/item is the new breaking item
        ratio = max(0.0, self.cap / i.weight)
        self.lb = self.value + ratio * i.value

    def add_item_incr(self, item: Item) -> float:
        item_idx = self.solution.problem.sorted_items_idx[item.ord]
        if item_idx < self.idx:
            return 0.0
        cap = self.cap
        value = self.value
        idx = self.idx
        # Go backwards
        cap -= item.weight
        value += item.value
        i = self.solution.problem.sorted_items[idx]
        while myfloat(cap) < 0:
            idx -= 1
            assert idx >= 0
            i = self.solution.problem.sorted_items[idx]
            if i.ord != item.ord and i.ord not in self.solution.used_set:
                cap += i.weight
                value -= i.value

        # idx/item is the new breaking item
        ratio = max(0.0, cap / i.weight)
        lb = value + ratio * i.value

        return lb - self.lb


# ---------------------------------- Solution --------------------------------


@final
class Solution(SupportsCopySolution, SupportsObjectiveValue[Float], SupportsLowerBound[Float]):
    def __init__(
        self,
        problem: Problem,
        used: list[Item],
        unused: list[Item],
        used_set: Optional[set[int]] = None,
        value: Optional[float] = None,
        cap: Optional[float] = None,
        lb: Optional[LowerBound] = None,
    ):
        self.problem = problem
        self.used = used
        self.unused = unused
        self.used_set = used_set if used_set is not None else set(map(lambda item: item.ord, self.used))
        self.value = value if value is not None else sum(map(lambda i: i.value, self.used))
        self.cap: float = cap if cap is not None else self.problem.capacity - sum(map(lambda i: i.weight, self.used))
        self._lb = lb

    @property
    def lb(self) -> LowerBound:
        if self._lb is None:
            self._lb = LowerBound.for_solution(self)
        return self._lb

    def __str__(self) -> str:
        return " ".join(map(lambda i: str(i + 1), self.used_set))

    def to_textio(self, f: TextIO) -> None:
        f.write(str(self) + "\n")

    def copy_solution(self) -> Self:
        sol = type(self)(
            problem=self.problem,
            used=self.used.copy(),
            unused=self.unused.copy(),
            value=self.value,
            cap=self.cap,
            used_set=self.used_set.copy(),
        )
        sol._lb = None if self._lb is None else self._lb.copy_to_solution(sol)
        return sol

    def objective_value(self) -> Optional[Float]:
        return myfloat(-self.value)

    def lower_bound(self) -> Float:
        return myfloat(-self.lb.lb)

    def add_item(self, ix: int) -> None:
        item = self.unused[ix]

        self.unused[ix] = self.unused[-1]
        self.unused.pop()
        self.used.append(item)
        self.used_set.add(item.ord)

        self.value += item.value
        self.cap -= item.weight

        if self._lb is not None:
            self._lb.add_item(item)

    def rem_item(self, ix: int) -> None:
        item = self.used[ix]

        self.used[ix] = self.used[-1]
        self.used.pop()
        self.used_set.remove(item.ord)
        self.unused.append(item)

        self.value -= item.value
        self.cap += item.weight

        self._lb = None


# ----------------------------------- Moves -----------------------------------


@final
class AddMove(
    SupportsApplyMove[Solution],
    SupportsLowerBoundIncrement[Solution, Float],
    SupportsObjectiveValueIncrement[Solution, Float],
):
    """
    Move to add a new item to the knapsack.
    """

    def __init__(self, ix: int):
        self.ix = ix

    def apply_move(self, solution: Solution) -> Solution:
        solution.add_item(self.ix)
        return solution

    def lower_bound_increment(self, solution: Solution) -> Float:
        item = solution.unused[self.ix]
        incr = solution.lb.add_item_incr(item)
        return myfloat(-incr)

    def objective_value_increment(self, solution: Solution) -> Float:
        item = solution.unused[self.ix]
        return myfloat(-item.value)


@final
class SwapMove(SupportsApplyMove[Solution], SupportsObjectiveValueIncrement[Solution, Float]):
    """
    Move to swap items in the knapsack. Removes `i` and inserts `j`
    """

    def __init__(self, ix: int, jx: int):
        self.ix = ix
        self.jx = jx

    def apply_move(self, solution: Solution) -> Solution:
        solution.rem_item(self.ix)
        solution.add_item(self.jx)

        assert myfloat(-solution.cap) <= 0

        return solution

    def objective_value_increment(self, solution: Solution) -> Float:
        iv = solution.used[self.ix].value
        jv = solution.unused[self.jx].value
        return myfloat(iv - jv)


# ------------------------------- Neighbourhood ------------------------------


@final
class AddNeighbourhood(
    SupportsMoves[Solution, AddMove],
    SupportsRandomMovesWithoutReplacement[Solution, AddMove],
    SupportsRandomMove[Solution, AddMove],
):
    def moves(self, solution: Solution) -> Iterable[AddMove]:
        for ix, i in enumerate(solution.unused):
            if myfloat(i.weight) <= solution.cap:
                yield AddMove(ix)

    def random_moves_without_replacement(self, solution: Solution) -> Iterable[AddMove]:
        for ix in sparse_fisher_yates_iter(len(solution.unused)):
            i = solution.unused[ix]
            if myfloat(i.weight) <= solution.cap:
                yield AddMove(ix)

    def random_move(self, solution: Solution) -> Optional[AddMove]:
        return next(iter(self.random_moves_without_replacement(solution)), None)


# FIXME: Implement remove neighbourhood


@final
class SwapNeighbourhood(
    SupportsMoves[Solution, SwapMove],
    SupportsRandomMovesWithoutReplacement[Solution, SwapMove],
    SupportsRandomMove[Solution, SwapMove],
):
    def moves(self, solution: Solution) -> Iterable[SwapMove]:
        for ix, i in enumerate(solution.used):
            for jx, j in enumerate(solution.unused):
                if myfloat(j.weight - i.weight) <= solution.cap:
                    yield SwapMove(ix, jx)

    def random_moves_without_replacement(self, solution: Solution) -> Iterable[SwapMove]:
        l1 = len(solution.used)
        l2 = len(solution.unused)
        for k in sparse_fisher_yates_iter(l1 * l2):
            ix, jx = divmod(k, l2)
            i = solution.used[ix]
            j = solution.unused[jx]
            if myfloat(j.weight - i.weight) <= solution.cap:
                yield SwapMove(ix, jx)

    def random_move(self, solution: Solution) -> Optional[SwapMove]:
        return next(iter(self.random_moves_without_replacement(solution)), None)


@final
class LocalNeighbourhood(
    SupportsMoves[Solution, AddMove | SwapMove],
    SupportsRandomMovesWithoutReplacement[Solution, AddMove | SwapMove],
    SupportsRandomMove[Solution, AddMove | SwapMove],
    SupportsSubNeighbourhoods[tuple[AddNeighbourhood, SwapNeighbourhood]],
):
    def moves(self, solution: Solution) -> Iterable[AddMove | SwapMove]:
        for ix, i in enumerate(solution.unused):
            if myfloat(i.weight) <= solution.cap:
                yield AddMove(ix)
        for ix, i in enumerate(solution.used):
            for jx, j in enumerate(solution.unused):
                if myfloat(j.weight - i.weight) <= solution.cap:
                    yield SwapMove(ix, jx)

    def random_moves_without_replacement(self, solution: Solution) -> Iterable[AddMove | SwapMove]:
        l1 = len(solution.used)
        l2 = len(solution.unused)
        nadd = l2
        nswap = l1 * l2
        for k in sparse_fisher_yates_iter(nswap + nadd):
            if k < nswap:
                ix, jx = divmod(k, l2)
                i = solution.used[ix]
                j = solution.unused[jx]
                if myfloat(j.weight - i.weight) <= solution.cap:
                    yield SwapMove(ix, jx)
            else:
                ix = k - nswap
                i = solution.unused[ix]
                if myfloat(i.weight) <= solution.cap:
                    yield AddMove(ix)

    def random_move(self, solution: Solution) -> Optional[AddMove | SwapMove]:
        return next(iter(self.random_moves_without_replacement(solution)), None)

    def sub_neighbourhoods(self) -> tuple[AddNeighbourhood, SwapNeighbourhood]:
        return (AddNeighbourhood(), SwapNeighbourhood())


# ---------------------------------- Problem --------------------------------


@final
class Problem(
    SupportsConstructionNeighbourhood[AddNeighbourhood],
    SupportsLocalNeighbourhood[LocalNeighbourhood],
    SupportsEmptySolution[Solution],
):
    def __init__(self, items: tuple[Item, ...], capacity: float):
        self.items = items
        self.capacity = capacity
        self.c_nbhood: Optional[AddNeighbourhood] = None
        self.l_nbhood: Optional[LocalNeighbourhood] = None
        self._sorted_items: Optional[tuple[Item, ...]] = None  # used for lower bound
        self._sorted_items_idx: Optional[dict[int, int]] = None  # used for lower bound

    @property
    def sorted_items(self) -> tuple[Item, ...]:
        if self._sorted_items is None:
            self._sorted_items = tuple(sorted(self.items, key=lambda i: i.value / i.weight))
        return self._sorted_items

    @property
    def sorted_items_idx(self) -> dict[int, int]:
        if self._sorted_items_idx is None:
            self._sorted_items_idx = {i.ord: ix for ix, i in enumerate(self.sorted_items)}
        return self._sorted_items_idx

    def construction_neighbourhood(self) -> AddNeighbourhood:
        if self.c_nbhood is None:
            self.c_nbhood = AddNeighbourhood()
        return self.c_nbhood

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
        capacity = float(f.readline().strip())
        items: list[Item] = []
        for i in range(n):
            v, w = map(float, f.readline().strip().split())
            items.append(Item(ord=i, value=v, weight=w))
        return cls(tuple(items), capacity)

    def empty_solution(self) -> Solution:
        return Solution(self, [], list(self.items))


if __name__ == "__main__":
    import roar_net_api.algorithms as alg

    logging.basicConfig(stream=sys.stderr, level="INFO", format="%(levelname)s;%(asctime)s;%(message)s")

    problem = Problem.from_textio(sys.stdin)

    # Run greedy construction to get an initial solution
    # solution = problem.empty_solution()
    solution = alg.greedy_construction(problem)
    # solution = alg.beam_search(problem, bw=10)
    # solution = alg.grasp(problem, 30.0)
    log.info(f"Objective value after constructive search: {solution.objective_value()}")

    # Run simulated annealing to improve the previous solution
    solution = alg.sa(problem, solution, 10.0, 50.0)
    # solution = alg.rls(problem, solution, 10.0)
    # solution = alg.best_improvement(problem, solution)
    # solution = alg.first_improvement(problem, solution)
    # solution = alg.vns_fi(problem, solution)
    log.info(f"Objective value after local search: {solution.objective_value()}")

    # Print the final solution to stdout
    solution.to_textio(sys.stdout)
