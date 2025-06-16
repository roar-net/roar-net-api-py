# SPDX-FileCopyrightText: © 2025 Authors of the roar-net-api-py project <https://github.com/roar-net/roar-net-api-py/blob/main/AUTHORS>
#
# SPDX-License-Identifier: Apache-2.0

from .apply_move import SupportsApplyMove
from .construction_neighbourhood import SupportsConstructionNeighbourhood
from .copy_solution import SupportsCopySolution
from .destruction_neighbourhood import SupportsDestructionNeighbourhood
from .empty_solution import SupportsEmptySolution
from .heuristic_solution import SupportsHeuristicSolution
from .invert_move import SupportsInvertMove
from .local_neighbourhood import SupportsLocalNeighbourhood
from .lower_bound import SupportsLowerBound
from .lower_bound_increment import SupportsLowerBoundIncrement
from .moves import SupportsMoves
from .objective_value import SupportsObjectiveValue
from .objective_value_increment import SupportsObjectiveValueIncrement
from .random_move import SupportsRandomMove
from .random_moves_without_replacement import SupportsRandomMovesWithoutReplacement
from .random_solution import SupportsRandomSolution
from .segments import (
    SupportsApplyMoveLeftEnd,
    SupportsApplyMoveRightEnd,
    SupportsRandomMovesRightEndFartherWithoutReplacement,
    SupportsSegment,
    SupportsSegmentLength,
)
from .sub_neighbourhoods import SupportsSubNeighbourhoods

__all__ = [
    "SupportsApplyMove",
    "SupportsConstructionNeighbourhood",
    "SupportsCopySolution",
    "SupportsDestructionNeighbourhood",
    "SupportsEmptySolution",
    "SupportsHeuristicSolution",
    "SupportsInvertMove",
    "SupportsLocalNeighbourhood",
    "SupportsLowerBoundIncrement",
    "SupportsLowerBound",
    "SupportsMoves",
    "SupportsObjectiveValueIncrement",
    "SupportsObjectiveValue",
    "SupportsRandomMove",
    "SupportsRandomMovesWithoutReplacement",
    "SupportsRandomSolution",
    "SupportsSegmentLength",
    "SupportsApplyMoveLeftEnd",
    "SupportsApplyMoveRightEnd",
    "SupportsSegment",
    "SupportsRandomMovesRightEndFartherWithoutReplacement",
    "SupportsSubNeighbourhoods",
]
