# SPDX-FileCopyrightText: © 2025 Authors of the roar-net-api-py project <https://github.com/roar-net/roar-net-api-py/blob/main/AUTHORS>
#
# SPDX-License-Identifier: Apache-2.0

import itertools
import random
from dataclasses import dataclass
from enum import Enum, auto

import pytest
from scipy import stats

from roar_net_api.algorithms.beam_search import KMin


class MovesOrder(Enum):
    OptFirstSeq = auto()
    NonOptFirstSeq = auto()
    OptFirstInterleaved = auto()
    NonOptFirstInterleaved = auto()
    Random = auto()


class KType(Enum):
    Less = auto()
    Equal = auto()
    Greater = auto()


@pytest.mark.parametrize("n, ktype, moves_order", list(itertools.product([1, 2, 5, 10], KType, MovesOrder)))
def test_kmin_uniform_tie_breaking(n: int, ktype: KType, moves_order: MovesOrder) -> None:
    """
    Tests that the KMin class does uniform tie breaking.
    """

    r = 100
    m = n * r

    @dataclass
    class Element:
        i: int
        opt: bool

        @property
        def key(self) -> int:
            return -2 if self.opt else -1

    def make_elements() -> list[Element]:
        match moves_order:
            case MovesOrder.OptFirstSeq:
                return [Element(i, i < n) for i in range(n * 2)]
            case MovesOrder.NonOptFirstSeq:
                return [Element(i, i < n) for i in itertools.chain(range(n, n * 2), range(n))]
            case MovesOrder.OptFirstInterleaved:
                return [m for i in range(n) for m in [Element(i, True), Element(n + i, False)]]
            case MovesOrder.NonOptFirstInterleaved:
                return [m for i in range(n) for m in [Element(n + i, False), Element(i, True)]]
            case MovesOrder.Random:
                return [Element(i, i < n) for i in random.sample(range(n * 2), n * 2)]

    elements = make_elements()

    def get_k() -> int:
        match ktype:
            case KType.Less:
                return (n + 1) // 2
            case KType.Equal:
                return n
            case KType.Greater:
                return n + (n + 1) // 2

    k = get_k()

    trials = 200
    rejections = 0
    for _ in range(trials):
        freq = [0] * n * 2
        for _ in range(m):
            kmin = KMin[int, Element](k, lambda el: el.key)
            for el in elements:
                kmin.insert(el)
            assert len(kmin) == k
            for el in kmin:
                freq[el.i] += 1

        if k < n:
            # In this case only optimal values are selected
            # uniformly. Not all can be selected (k < n), but every
            # item should appear a similar number of times.
            assert freq[n:] == [0] * n
            if stats.chisquare(freq[:n]).pvalue < 0.001:
                rejections += 1
        elif k == n:
            # In this case only optimal values are selected and they
            # are all selected all the time.
            assert freq == ([m] * n) + ([0] * n)
        else:
            # In this case optimal values are always all selected, and
            # non-optimal are selected uniformly.
            assert freq[:n] == [m] * n
            if stats.chisquare(freq[n:]).pvalue < 0.001:
                rejections += 1

    if k != n:
        assert rejections / trials < 0.05
