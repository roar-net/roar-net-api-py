# SPDX-FileCopyrightText: © 2025 Authors of the roar-net-api-py project <https://github.com/roar-net/roar-net-api-py/blob/main/AUTHORS>
#
# SPDX-License-Identifier: Apache-2.0

import itertools
import random
from dataclasses import dataclass
from enum import Enum, auto
from typing import final

import pytest
from scipy import stats

from roar_net_api.algorithms import greedy_construction
from roar_net_api.operations import (
    SupportsApplyMove,
    SupportsConstructionNeighbourhood,
    SupportsEmptySolution,
    SupportsLowerBoundIncrement,
    SupportsMoves,
)


class MovesOrder(Enum):
    OptFirstSeq = auto()
    NonOptFirstSeq = auto()
    OptFirstInterleaved = auto()
    NonOptFirstInterleaved = auto()
    Random = auto()


@pytest.mark.parametrize("n, moves_order", list(itertools.product([1, 2, 5, 10], MovesOrder)))
def test_uniform_tie_breaking(n: int, moves_order: MovesOrder) -> None:
    """
    Tests that the greedy_construction method selects the best
    solution uniformly when doing tie breaking.

    Builds a problem that returns n equally optimal moves (and another
    n non-optimal) then runs greedy constructions m = n * r times. We
    expect greedy construction to select optimal moves with a
    frequency of about r for each each optimal move and exactly 0
    non-optimal moves.

    We use the chi-square test to test for uniformity. We run multiple
    trials of the test (with new sampled data each time) with a
    p-value of 0.001 and then if the number of rejected trials is more
    than 5% we fail the assertion. Note that, the expected rejection
    rate is around 0.1% (same as p-value). However, being too strict
    could lead to false negatives in CI. As such we use a much higher
    value (5%) to be conservative and avoid false negatives. This
    should be safe for our purposes since mistakes in the reservoir
    sampling algorithm should break the uniformity more often than not
    so we would expect many rejections (more than 5%).
    """

    r = 100
    m = n * r

    @dataclass
    @final
    class Solution:
        move: int | None = None

    @dataclass
    @final
    class Move(SupportsApplyMove[Solution], SupportsLowerBoundIncrement[Solution]):
        i: int
        opt: bool

        def apply_move(self, solution: Solution) -> Solution:
            solution.move = self.i
            return solution

        def lower_bound_increment(self, solution: Solution) -> int:
            return 0 if self.opt else 1

    def make_moves() -> list[Move]:
        match moves_order:
            case MovesOrder.OptFirstSeq:
                return [Move(i, i < n) for i in range(n * 2)]
            case MovesOrder.NonOptFirstSeq:
                return [Move(i, i < n) for i in itertools.chain(range(n, n * 2), range(n))]
            case MovesOrder.OptFirstInterleaved:
                return [m for i in range(n) for m in [Move(i, True), Move(n + i, False)]]
            case MovesOrder.NonOptFirstInterleaved:
                return [m for i in range(n) for m in [Move(n + i, False), Move(i, True)]]
            case MovesOrder.Random:
                return [Move(i, i < n) for i in random.sample(range(n * 2), n * 2)]

    # Generate the moves order only once for the whole test. This
    # can't be inside the neighbourhood moves method itself because of
    # the random case. In particular, if we have a different random
    # order each time, the constructions will be uniform if the moves
    # sampling is also uniform. For the other cases, it is also faster
    # to generate all the moves once, and there is no benefit in doing
    # the generation from scratch every time.
    moves = make_moves()

    @final
    class Neighbourhood(SupportsMoves[Solution, Move]):
        def moves(self, solution: Solution) -> list[Move]:
            if solution.move is None:
                return moves
            else:
                return []

    @final
    class Problem(
        SupportsConstructionNeighbourhood[Neighbourhood],
        SupportsEmptySolution[Solution],
    ):
        def construction_neighbourhood(self) -> Neighbourhood:
            return Neighbourhood()

        def empty_solution(self) -> Solution:
            return Solution()

    trials = 200
    rejections = 0
    for _ in range(trials):
        freq = [0] * n
        prob = Problem()
        for _ in range(m):
            s = greedy_construction(prob)
            assert s.move is not None
            freq[s.move] += 1

        if n == 1:
            if freq != [r] * n:
                rejections += 1
        else:
            if stats.chisquare(freq).pvalue < 0.001:
                rejections += 1

    if n == 1:
        assert rejections == 0
    else:
        assert rejections / trials < 0.05
