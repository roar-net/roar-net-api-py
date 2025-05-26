# SPDX-FileCopyrightText: © 2025 Authors of the roar-net-api-py project <https://github.com/roar-net/roar-net-api-py/blob/main/AUTHORS>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Protocol, TypeVar, Optional

Solution = TypeVar("Solution", contravariant=True)
Move = TypeVar("Move", covariant=True)


class SupportsRandomMove(Protocol[Solution, Move]):
    def random_move(self, solution: Solution) -> Optional[Move]: ...
