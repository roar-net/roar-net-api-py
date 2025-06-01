# SPDX-FileCopyrightText: © 2025 Authors of the roar-net-api-py project <https://github.com/roar-net/roar-net-api-py/blob/main/AUTHORS>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Protocol

from ..operations import (
    SupportsConstructionNeighbourhood,
    SupportsDestructionNeighbourhood,
    SupportsEmptySolution,
    SupportsHeuristicSolution,
    SupportsLocalNeighbourhood,
    SupportsRandomSolution,
)


class Problem(
    SupportsConstructionNeighbourhood,
    SupportsDestructionNeighbourhood,
    SupportsEmptySolution,
    SupportsHeuristicSolution,
    SupportsLocalNeighbourhood,
    SupportsRandomSolution,
    Protocol,
): ...
