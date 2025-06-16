#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: © 2025 Authors of the roar-net-api-py project <https://github.com/roar-net/roar-net-api-py/blob/main/AUTHORS>
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import math
import random
import sys
from collections.abc import Iterable, Sequence
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
    SupportsRandomSolution,
    SupportsSegmentLength,
    SupportsApplyMoveLeftEnd,
    SupportsApplyMoveRightEnd,
    SupportsSegment,
    SupportsRandomMovesRightEndFartherWithoutReplacement,
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


# ---------------------------------- Solution --------------------------------


@final
class Solution(SupportsCopySolution, SupportsObjectiveValue[int], SupportsLowerBound[int]):
    def __init__(self, problem: Problem, tour: list[int], not_visited: set[int], lb: int):
        self.problem = problem
        self.tour = tour
        self.not_visited = not_visited
        self.lb = lb

    def __str__(self) -> str:
        return " ".join(map(str, self.tour))

    @property
    def is_feasible(self) -> bool:
        return len(self.not_visited) == 0

    def to_textio(self, f: TextIO) -> None:
        f.write("NAME : %s\nTYPE : TOUR\n" % (self.problem.name + ".tour"))
        f.write("DIMENSION : %d\nTOUR_SECTION\n" % self.problem.n)
        f.write("\n".join(map(lambda x: str(x + 1), self.tour)))
        f.write("\nEOF\n")

    def copy_solution(self) -> Self:
        return self.__class__(self.problem, self.tour.copy(), self.not_visited.copy(), self.lb)

    def objective_value(self) -> Optional[int]:
        if self.is_feasible:
            return self.lb
        return None

    def lower_bound(self) -> int:
        return self.lb


# ----------------------------------- Moves -----------------------------------


@final
class AddMove(SupportsApplyMove[Solution], SupportsLowerBoundIncrement[Solution, int]):
    def __init__(self, neighbourhood: AddNeighbourhood, i: int, j: int):
        self.neighbourhood = neighbourhood
        # i and j are cities
        self.i = i
        self.j = j

    def apply_move(self, solution: Solution) -> Solution:
        assert solution.tour[-1] == self.i
        prob = solution.problem
        # Update lower bound
        solution.lb += prob.dist[self.i][self.j]
        if len(solution.not_visited) == 1:
            solution.lb += prob.dist[self.j][solution.tour[0]]
        # Tighter, but *not* better!
        # solution.lb += prob.dist[self.j][solution.tour[0]] - prob.dist[self.i][solution.tour[0]]
        # Update solution
        solution.tour.append(self.j)
        solution.not_visited.remove(self.j)
        return solution

    def lower_bound_increment(self, solution: Solution) -> int:
        assert solution.tour[-1] == self.i
        prob = solution.problem
        incr = prob.dist[self.i][self.j]
        if len(solution.not_visited) == 1:
            incr += prob.dist[self.j][solution.tour[0]]
        # Tighter, but *not* better!
        # incr += prob.dist[self.j][solution.tour[0]] - prob.dist[self.i][solution.tour[0]]
        return incr


@final
class TwoOptMove(
    SupportsApplyMove[Solution],
    SupportsObjectiveValueIncrement[Solution, int],
    SupportsApplyMoveLeftEnd["BreakpointSegment"],
    SupportsApplyMoveRightEnd["BreakpointSegment"],
):
    def __init__(self, neighbourhood: TwoOptNeighbourhood, ix: int, jx: int):
        self.neighbourhood = neighbourhood
        # ix and jx are indices
        self.ix = ix
        self.jx = jx

    def apply_move(self, solution: Solution) -> Solution:
        prob = solution.problem
        n, ix, jx = prob.n, self.ix, self.jx
        # Update tour length
        t = solution.tour
        solution.lb -= prob.dist[t[ix - 1]][t[ix]] + prob.dist[t[jx - 1]][t[jx % n]]
        solution.lb += prob.dist[t[ix - 1]][t[jx - 1]] + prob.dist[t[ix]][t[jx % n]]
        # Update solution
        solution.tour[ix:jx] = solution.tour[ix:jx][::-1]
        return solution

    def objective_value_increment(self, solution: Solution) -> int:
        prob = solution.problem
        n, ix, jx = prob.n, self.ix, self.jx
        # Tour length increment
        t = solution.tour
        incr = prob.dist[t[ix - 1]][t[jx - 1]] + prob.dist[t[ix]][t[jx % n]]
        incr -= prob.dist[t[ix - 1]][t[ix]] + prob.dist[t[jx - 1]][t[jx % n]]
        return incr

    def apply_move_left_end(self, segment: BreakpointSegment) -> BreakpointSegment:
        prob = segment.neighbourhood.problem
        n, ix, jx = prob.n, self.ix, self.jx
        # Update segment
        t0 = segment.t0
        t1 = segment.t1
        t0[ix:jx] = t0[ix:jx][::-1]
        # FIXME: Can tt and the length be updated more efficiently?
        t0i = n * [-1]
        for i in range(n):
            t0i[t0[i]] = i
        assert all(map(lambda x: x != -1, t0i))
        # Transformed permutation
        tt = segment.tt = [t0i[t1[i]] for i in range(n)]
        segment.length = sum((1 < abs(tt[i] - tt[i - 1]) < n - 1) for i in range(n))
        return segment

    def apply_move_right_end(self, segment: BreakpointSegment) -> BreakpointSegment:
        prob = segment.neighbourhood.problem
        n, ix, jx = prob.n, self.ix, self.jx
        # Update segment
        t1 = segment.t1
        tt = segment.tt
        t1[ix:jx] = t1[ix:jx][::-1]
        segment.length -= (1 < abs(tt[ix] - tt[ix - 1]) < n - 1) + (1 < abs(tt[jx % n] - tt[jx - 1]) < n - 1)
        tt[ix:jx] = tt[ix:jx][::-1]
        segment.length += (1 < abs(tt[ix] - tt[ix - 1]) < n - 1) + (1 < abs(tt[jx % n] - tt[jx - 1]) < n - 1)
        return segment


# -------------------------------- Segment -----------------------------------


@final
class BreakpointSegment(SupportsSegmentLength):
    def __init__(self, neighbourhood: TwoOptNeighbourhood, t0: list[int], t1: list[int], tt: list[int], length: int):
        self.neighbourhood = neighbourhood
        self.t0 = t0  # Left end tour
        self.t1 = t1  # Right end tour
        self.tt = tt  # Transformed tour to make breakpoint detection simpler
        self.length = length

    def segment_length(self) -> int:
        assert self.length == sum(
            (1 < abs(self.tt[i] - self.tt[i - 1]) < len(self.tt) - 1) for i in range(len(self.tt))
        )
        return self.length


# ------------------------------- Neighbourhood ------------------------------


@final
class AddNeighbourhood(SupportsMoves[Solution, AddMove]):
    def __init__(self, problem: Problem):
        self.problem = problem

    def moves(self, solution: Solution) -> Iterable[AddMove]:
        assert self.problem == solution.problem
        i = solution.tour[-1]
        for j in solution.not_visited:
            yield AddMove(self, i, j)


@final
class TwoOptNeighbourhood(
    SupportsMoves[Solution, TwoOptMove],
    SupportsRandomMovesWithoutReplacement[Solution, TwoOptMove],
    SupportsRandomMove[Solution, TwoOptMove],
    SupportsSegment[Solution, BreakpointSegment],
    SupportsRandomMovesRightEndFartherWithoutReplacement[BreakpointSegment, TwoOptMove],
):
    def __init__(self, problem: Problem):
        self.problem = problem

    def moves(self, solution: Solution) -> Iterable[TwoOptMove]:
        assert self.problem == solution.problem
        n = self.problem.n
        # This is only meant to be used as a local neighbourhood, so solution should be feasible
        assert solution.is_feasible
        for ix in range(1, n - 1):
            for jx in range(ix + 2, n + (ix != 1)):
                yield TwoOptMove(self, ix, jx)

    def random_moves_without_replacement(self, solution: Solution) -> Iterable[TwoOptMove]:
        assert self.problem == solution.problem
        n = self.problem.n
        # This is only meant to be used as a local neighbourhood, so solution should be feasible
        assert solution.is_feasible
        # Sample integers at random and convert them into moves. To that
        # end, start by mapping x = 0, 1, ..., onto pairs (a, b) as shown
        # in the following table:
        #
        #    b  0   1   2   3   4   5
        #  a +------------------------
        #  0 |  -   -   -   -   -   -
        #  1 |  0   -   -   -   -   -
        #  2 |  1   2   -   -   -   -
        #  3 |  3   4   5   -   -   -
        #  4 |  6   7   8   9   -   -
        #  5 | 10  11  12  13  14   -
        #  6 | 15   …   …   …   …   …
        #
        # Note how x = a*(a-1)/2 + b.
        # To solve for a given x, rewrite the expression as
        # a**2 - a + 2*b - 2*x = 0, which has one positive root:
        # a = (1 + sqrt(1 + 8*x - 8*b) / 2
        # Taking a = floor((1 + sqrt(1 + 8*x)) / 2) and
        # b = x - a*(a-1)/2 allows both the desired jx = a + 2 and
        # ix = b + 1 to be obtained.
        # Note: since pair (1, n) would not be a valid 2-opt move, it can
        # be skipped or simply replaced by (n-2, n) when generated, which
        # saves one iteration.
        for x in sparse_fisher_yates_iter(n * (n - 3) // 2):
            jx = (1 + math.isqrt(1 + 8 * x)) // 2
            ix = x - jx * (jx - 1) // 2 + 1
            jx += 2
            # Handle special case
            if ix == 1 and jx == n:
                ix = n - 2
            yield TwoOptMove(self, ix, jx)

    def random_move(self, solution: Solution) -> Optional[TwoOptMove]:
        return next(iter(self.random_moves_without_replacement(solution)), None)

    def segment(self, solution0: Solution, solution1: Solution) -> Optional[BreakpointSegment]:
        assert self.problem == solution0.problem == solution1.problem
        t0 = solution0.tour
        t1 = solution1.tour
        n = self.problem.n
        if n == len(t0) and n == len(t1):  # Both solutions are feasible
            # Inverse permutation of t0
            t0i = n * [-1]
            for i in range(n):
                t0i[t0[i]] = i
            assert all(map(lambda x: x != -1, t0i))
            # Transformed permutation
            tt = [t0i[t1[i]] for i in range(n)]
            length = sum((1 < abs(tt[i] - tt[i - 1]) < n - 1) for i in range(n))
            return BreakpointSegment(self, t0.copy(), t1.copy(), tt, length)
        return None

    def moves_right_end_closer(self, segment: BreakpointSegment) -> Iterable[TwoOptMove]:
        tt = segment.tt
        n = self.problem.n
        # FIXME: Brute force approach, may be very inefficient
        # FIXME: There may be no 2-opt moves that decrease the breakpoint
        #        distance even if it is not 0
        for ix in range(1, n - 1):
            for jx in range(ix + 2, n + (ix != 1)):
                if (1 < abs(tt[jx - 1] - tt[ix - 1]) < n - 1) + (1 < abs(tt[jx % n] - tt[ix]) < n - 1) < (
                    1 < abs(tt[ix] - tt[ix - 1]) < n - 1
                ) + (1 < abs(tt[jx % n] - tt[jx - 1]) < n - 1):
                    yield TwoOptMove(self, ix, jx)

    def moves_right_end_farther(self, segment: BreakpointSegment) -> Iterable[TwoOptMove]:
        tt = segment.tt
        n = self.problem.n
        for ix in range(1, n - 1):
            for jx in range(ix + 2, n + (ix != 1)):
                if (1 < abs(tt[jx - 1] - tt[ix - 1]) < n - 1) + (1 < abs(tt[jx % n] - tt[ix]) < n - 1) > (
                    1 < abs(tt[ix] - tt[ix - 1]) < n - 1
                ) + (1 < abs(tt[jx % n] - tt[jx - 1]) < n - 1):
                    yield TwoOptMove(self, ix, jx)

    def random_moves_right_end_closer_without_replacement(self, segment: BreakpointSegment) -> Iterable[TwoOptMove]:
        tt = segment.tt
        n = self.problem.n
        # FIXME: Brute force approach, may be very inefficient
        # FIXME: There may be no 2-opt moves that decrease the breakpoint
        #        distance even if it is not 0
        for x in sparse_fisher_yates_iter(n * (n - 3) // 2):
            jx = (1 + math.isqrt(1 + 8 * x)) // 2
            ix = x - jx * (jx - 1) // 2 + 1
            jx += 2
            # Handle special case
            if ix == 1 and jx == n:
                ix = n - 2
            if (1 < abs(tt[jx - 1] - tt[ix - 1]) < n - 1) + (1 < abs(tt[jx % n] - tt[ix]) < n - 1) < (
                1 < abs(tt[ix] - tt[ix - 1]) < n - 1
            ) + (1 < abs(tt[jx % n] - tt[jx - 1]) < n - 1):
                yield TwoOptMove(self, ix, jx)

    def random_moves_right_end_farther_without_replacement(self, segment: BreakpointSegment) -> Iterable[TwoOptMove]:
        tt = segment.tt
        n = self.problem.n
        for x in sparse_fisher_yates_iter(n * (n - 3) // 2):
            jx = (1 + math.isqrt(1 + 8 * x)) // 2
            ix = x - jx * (jx - 1) // 2 + 1
            jx += 2
            # Handle special case
            if ix == 1 and jx == n:
                ix = n - 2
            if (1 < abs(tt[jx - 1] - tt[ix - 1]) < n - 1) + (1 < abs(tt[jx % n] - tt[ix]) < n - 1) > (
                1 < abs(tt[ix] - tt[ix - 1]) < n - 1
            ) + (1 < abs(tt[jx % n] - tt[jx - 1]) < n - 1):
                yield TwoOptMove(self, ix, jx)

    def random_move_right_end_closer(self, segment: BreakpointSegment) -> Optional[TwoOptMove]:
        return next(iter(self.random_moves_right_end_closer_without_replacement(segment)), None)

    def random_move_right_end_farther(self, segment: BreakpointSegment) -> Optional[TwoOptMove]:
        return next(iter(self.random_moves_right_end_farther_without_replacement(segment)), None)


# ---------------------------------- Problem --------------------------------


@final
class Problem(
    SupportsConstructionNeighbourhood[AddNeighbourhood],
    SupportsLocalNeighbourhood[TwoOptNeighbourhood],
    SupportsEmptySolution[Solution],
    SupportsRandomSolution[Solution],
):
    def __init__(self, dist: tuple[tuple[int, ...], ...], name: str):
        self.dist = tuple(tuple(t) for t in dist)
        self.name = name
        self.n = len(self.dist)
        self.c_nbhood: Optional[AddNeighbourhood] = None
        self.l_nbhood: Optional[TwoOptNeighbourhood] = None

    def __str__(self) -> str:
        out: list[str] = []
        for row in self.dist:
            out.append(" ".join(map(str, row)))
        return "\n".join(out)

    def construction_neighbourhood(self) -> AddNeighbourhood:
        if self.c_nbhood is None:
            self.c_nbhood = AddNeighbourhood(self)
        return self.c_nbhood

    def local_neighbourhood(self) -> TwoOptNeighbourhood:
        if self.l_nbhood is None:
            self.l_nbhood = TwoOptNeighbourhood(self)
        return self.l_nbhood

    @classmethod
    def from_textio(cls, f: TextIO) -> Self:
        """
        Create a problem from a text I/O source `f` in TSPLIB format
        """
        s = f.readline().strip()
        n = None
        dt = None
        name = "unnamed"
        while s != "NODE_COORD_SECTION" and s != "":
            line = s.split(":", 1)
            k = line[0].strip()
            if k == "DIMENSION":
                n = int(line[1])
            elif k == "EDGE_WEIGHT_TYPE":
                dt = line[1].strip()
            elif k == "NAME":
                name = line[1].strip()
            s = f.readline().strip()
        if n is not None and dt == "EUC_2D":
            kxy: list[tuple[float, ...]] = []
            for i in range(n):
                kxy.append(tuple(map(float, f.readline().split())))
            kxy = sorted(kxy)
            dist: list[tuple[int, ...]] = []
            for i in range(n):
                if kxy[i][0] != i + 1:
                    raise Exception("Invalid instance")
                aux: list[int] = []
                for j in range(n):
                    aux.append(int(0.5 + math.sqrt((kxy[i][1] - kxy[j][1]) ** 2 + (kxy[i][2] - kxy[j][2]) ** 2)))
                dist.append(tuple(aux))
            return cls(tuple(dist), name)
        else:
            raise Exception(f"Instance format {dt} not supported")

    def empty_solution(self) -> Solution:
        return Solution(self, [0], set(range(1, self.n)), 0)

    def random_solution(self) -> Solution:
        c = list(range(1, self.n))
        random.shuffle(c)
        c.insert(0, 0)
        obj = self.dist[c[-1]][c[0]]
        for ix in range(1, self.n):
            obj += self.dist[c[ix - 1]][c[ix]]
        return Solution(self, c, set(), obj)


if __name__ == "__main__":
    import roar_net_api.algorithms as alg

    logging.basicConfig(stream=sys.stderr, level="INFO", format="%(levelname)s;%(asctime)s;%(message)s")

    problem = Problem.from_textio(sys.stdin)

    # Run greedy construction to get an initial solution
    solution = alg.greedy_construction(problem)
    # solution = alg.beam_search(problem, bw=10)
    # solution = alg.grasp(problem, 30.0)
    log.info(f"Objective value after constructive search: {solution.objective_value()}")

    # Run simulated annealing to improve the previous solution
    # solution = alg.sa(problem, solution, 10.0, 30.0)
    # solution = alg.rls(problem, solution, 10.0)
    # solution = alg.best_improvement(problem, solution)
    # solution = alg.first_improvement(problem, solution)
    solution = alg.tabu_search(problem, solution, 10.0, 200)  # type: ignore # FIXME: currently typing not working
    log.info(f"Objective value after local search: {solution.objective_value()}")

    # Print the final solution to stdout
    solution.to_textio(sys.stdout)
