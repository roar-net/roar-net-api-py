# SPDX-FileCopyrightText: © 2025 Authors of the roar-net-api-py project <https://github.com/roar-net/roar-net-api-py/blob/main/AUTHORS>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Protocol, TypeVar

from ..operations import (
    SupportsMoves,
    SupportsRandomMove,
    SupportsRandomMovesWithoutReplacement,
)

_MoveT = TypeVar("_MoveT", covariant=True)
_SolutionT = TypeVar("_SolutionT", contravariant=True)


class Neighbourhood(
    SupportsMoves[_SolutionT, _MoveT],
    SupportsRandomMovesWithoutReplacement[_SolutionT, _MoveT],
    SupportsRandomMove[_SolutionT, _MoveT],
    Protocol,
): ...
