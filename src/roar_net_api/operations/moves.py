# SPDX-FileCopyrightText: © 2025 Authors of the roar-net-api-py project <https://github.com/roar-net/roar-net-api-py/blob/main/AUTHORS>
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable
from typing import Protocol, TypeVar

Solution = TypeVar("Solution", contravariant=True)
Move = TypeVar("Move", covariant=True)


class SupportsMoves(Protocol[Solution, Move]):
    def moves(self, solution: Solution) -> Iterable[Move]: ...
