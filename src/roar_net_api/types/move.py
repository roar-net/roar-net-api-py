# SPDX-FileCopyrightText: © 2025 Authors of the roar-net-api-py project <https://github.com/roar-net/roar-net-api-py/blob/main/AUTHORS>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Protocol, TypeVar

from ..operations import (
    SupportsApplyMove,
    SupportsRevertMove,
    SupportsLowerBoundIncrement,
    SupportsObjectiveValueIncrement,
)

_SolutionT = TypeVar("_SolutionT")


class Move(
    SupportsApplyMove[_SolutionT],
    SupportsRevertMove[_SolutionT],
    SupportsLowerBoundIncrement[_SolutionT],
    SupportsObjectiveValueIncrement[_SolutionT],
    Protocol,
): ...
