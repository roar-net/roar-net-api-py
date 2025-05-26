# SPDX-FileCopyrightText: © 2025 Authors of the roar-net-api-py project <https://github.com/roar-net/roar-net-api-py/blob/main/AUTHORS>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Protocol, TypeVar
from collections.abc import Iterable

Solution = TypeVar("Solution", contravariant=True)
Move = TypeVar("Move", covariant=True)


class SupportsRandomMovesWithoutReplacement(Protocol[Solution, Move]):
    def random_moves_without_replacement(self, solution: Solution) -> Iterable[Move]: ...
