#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: © 2025 Authors of the roar-net-api-py project <https://github.com/roar-net/roar-net-api-py/blob/main/AUTHORS>
#
# SPDX-License-Identifier: Apache-2.0

import logging
import math
import random
import sys
from collections.abc import Iterable
from logging import getLogger
from typing import Optional, Self, TextIO, final

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
)

log = getLogger(__name__)


def argmin(seq):
    return min(range(len(seq)), key=seq.__getitem__)


def sparse_fisher_yates_iter(n):
    p = dict()
    for i in range(n - 1, -1, -1):
        r = random.randrange(i + 1)
        yield p.get(r, r)
        if i != r:
            # p[r] = p.pop(i, i) # saves memory, takes time
            p[r] = p.get(i, i)  # lazy, but faster


# ---------------------------------- Solution --------------------------------


@final
class Solution(SupportsCopySolution, SupportsObjectiveValue, SupportsLowerBound):
    def __init__(self, problem, tour, not_visited, lb):
        self.problem = problem
        self.tour = tour
        self.not_visited = not_visited
        self.lb = lb

    def __str__(self):
        return " ".join(map(str, self.tour))

    @property
    def is_feasible(self):
        return len(self.not_visited) == 0

    def to_textio(self, f: TextIO):
        f.write("NAME : %s\nTYPE : TOUR\n" % (self.problem.name + ".tour"))
        f.write("DIMENSION : %d\nTOUR_SECTION\n" % len(self.tour))
        f.write("\n".join(map(lambda x: str(x+1), self.tour)))
        f.write("\nEOF\n")

    def copy_solution(self) -> Self:
        return self.__class__(self.problem, self.tour.copy(), self.not_visited.copy(), self.lb)

    def objective_value(self) -> Optional[float]:
        if self.is_feasible:
            return self.lb
        return None

    def lower_bound(self) -> float:
        return self.lb


# ----------------------------------- Moves -----------------------------------


@final
class AddMove(SupportsApplyMove[Solution], SupportsLowerBoundIncrement[Solution]):
    def __init__(self, neighbourhood, i, j):
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

    def lower_bound_increment(self, solution: Solution) -> float:
        assert solution.tour[-1] == self.i
        prob = solution.problem
        incr = prob.dist[self.i][self.j]
        if len(solution.not_visited) == 1:
            incr += prob.dist[self.j][solution.tour[0]]
        # Tighter, but *not* better!
        # incr += prob.dist[self.j][solution.tour[0]] - prob.dist[self.i][solution.tour[0]]
        return incr


@final
class TwoOptMove(SupportsApplyMove[Solution], SupportsObjectiveValueIncrement[Solution]):
    def __init__(self, neighbourhood, ix, jx):
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

    def objective_value_increment(self, solution: Solution) -> float:
        prob = solution.problem
        n, ix, jx = prob.n, self.ix, self.jx
        # Tour length increment
        t = solution.tour
        incr = prob.dist[t[ix - 1]][t[jx - 1]] + prob.dist[t[ix]][t[jx % n]]
        incr -= prob.dist[t[ix - 1]][t[ix]] + prob.dist[t[jx - 1]][t[jx % n]]
        return incr


# ------------------------------- Neighbourhood ------------------------------


class Neighbourhood:
    def __init__(self, problem):
        self.problem = problem


@final
class AddNeighbourhood(Neighbourhood, SupportsMoves[Solution, AddMove]):
    def moves(self, solution: Solution) -> Iterable[AddMove]:
        assert self.problem == solution.problem
        i = solution.tour[-1]
        for j in solution.not_visited:
            yield AddMove(self, i, j)


@final
class TwoOptNeighbourhood(
    Neighbourhood,
    SupportsMoves[Solution, TwoOptMove],
    SupportsRandomMovesWithoutReplacement[Solution, TwoOptMove],
    SupportsRandomMove[Solution, TwoOptMove],
):
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


# ---------------------------------- Problem --------------------------------


@final
class Problem(
    SupportsConstructionNeighbourhood[AddNeighbourhood],
    SupportsLocalNeighbourhood[TwoOptNeighbourhood],
    SupportsEmptySolution[Solution],
    SupportsRandomSolution[Solution],
):
    def __init__(self, dist, name):
        self.dist = tuple(tuple(t) for t in dist)
        self.name = name
        self.n = len(self.dist)
        self.c_nbhood = None
        self.l_nbhood = None

    def __str__(self):
        out = [str(self.n)]
        for i, j in self.xy:
            out.append("%d %d" % (i, j))
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
    def from_textio(cls, f):
        """
        Create a problem from a text I/O source `f` in TSPLIB format
        """
        s = f.readline().strip()
        n = dt = None
        name = "unamed"
        while s != "NODE_COORD_SECTION" and s != "":
            l = s.split(":", 1)
            k = l[0].strip()
            if k == "DIMENSION":
                n = int(l[1])
            elif k == "EDGE_WEIGHT_TYPE":
                dt = l[1].strip()
            elif k == "NAME":
                name = l[1].strip()
            s = f.readline().strip()
        if n is not None and dt == "EUC_2D":
            kxy = []
            for i in range(n):
                kxy.append(tuple(map(float, f.readline().split())))
            kxy = sorted(kxy)
            dist = []
            for i in range(n):
                if kxy[i][0] != i+1:
                    print("ERROR: Invalid instance")
                    return None
                aux = []
                for j in range(n):
                    aux.append(int(0.5 + math.sqrt((kxy[i][1] - kxy[j][1]) ** 2 + (kxy[i][2] - kxy[j][2]) ** 2)))
                dist.append(tuple(aux))
            dist = tuple(dist)
            return cls(dist, name)
        else:
            print("ERROR: Instance format not supported")

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
    solution = alg.sa(problem, solution, 10.0, 30.0)
    # solution = alg.rls(problem, solution, 10.0)
    # solution = alg.best_improvement(problem, solution)
    # solution = alg.first_improvement(problem, solution)
    log.info(f"Objective value after local search: {solution.objective_value()}")

    # Print the final solution to stdout
    print(str(solution))
    solution.to_textio(sys.stdout)
