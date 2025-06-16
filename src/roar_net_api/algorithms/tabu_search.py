#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: © 2025 Authors of the roar-net-api-py project <https://github.com/roar-net/roar-net-api-py/blob/main/AUTHORS>
#
# SPDX-License-Identifier: Apache-2.0

from logging import getLogger
from time import perf_counter
from typing import Protocol, TypeVar

from ..operations import (
    SupportsCopySolution,
    SupportsObjectiveValue,
    SupportsLocalNeighbourhood,
    SupportsApplyMove,
    SupportsObjectiveValueIncrement,
    SupportsSegmentLength,
    SupportsApplyMoveLeftEnd,
    SupportsApplyMoveRightEnd,
    SupportsSegment,
    SupportsRandomMovesRightEndFartherWithoutReplacement,
)
from ..values import Float

log = getLogger(__name__)

_Value = int | float | Float


class _Solution(SupportsObjectiveValue[_Value], SupportsCopySolution, Protocol): ...


_TSolution = TypeVar("_TSolution", bound=_Solution)


class _Segment(SupportsSegmentLength, Protocol): ...


class _Move(
    SupportsApplyMove[_TSolution],
    SupportsObjectiveValueIncrement[_TSolution, _Value],
    SupportsApplyMoveLeftEnd[_Segment],
    SupportsApplyMoveRightEnd[_Segment],
    Protocol[_TSolution],
): ...


class _Neighbourhood(
    SupportsSegment[_TSolution, _Segment],
    SupportsRandomMovesRightEndFartherWithoutReplacement[_Segment, _Move[_TSolution]],
    Protocol[_TSolution],
): ...


class _Problem(SupportsLocalNeighbourhood[_Neighbourhood[_TSolution]], Protocol): ...


def tabu_search(problem: _Problem[_TSolution], solution: _TSolution, budget: float, ntabu: int = 5) -> _TSolution:
    # modifies solution in place and returns a reference to it
    start = perf_counter()
    nbhood = problem.local_neighbourhood()
    seg = nbhood.segment(solution, solution)
    assert seg is not None
    best = solution.copy_solution()
    best_obj = solution.objective_value()
    assert best_obj is not None
    recent_moves: list[_Move[_TSolution]] = []
    move_iter = iter(nbhood.random_moves_right_end_farther_without_replacement(seg))
    move = next(move_iter, None)
    while move is not None and perf_counter() - start < budget:
        best_move = move
        best_incr = move.objective_value_increment(solution)
        assert best_incr is not None
        for move in move_iter:
            incr = move.objective_value_increment(solution)
            assert incr is not None
            if incr < best_incr:  # type: ignore
                best_move, best_incr = move, incr
            if best_incr < 0:
                break
        # accept best move even if it degrades the solution
        recent_moves.append(best_move)
        solution = best_move.apply_move(solution)
        seg = best_move.apply_move_right_end(seg)
        obj = solution.objective_value()
        assert obj is not None
        if obj < best_obj:  # type: ignore
            best = solution.copy_solution()
            best_obj = obj
            log.info(f"Best solution: {solution.objective_value()}")
            log.debug(f"Length recent moves: {len(recent_moves)}; Segment length: {seg.segment_length()}")
            # Clear tabu list
            seg = nbhood.segment(solution, solution)
            assert seg is not None
            recent_moves.clear()
        # FIXME: For edit distances, len(recent_moves) == seg.length(),
        #        but not for other distances. Which condition is best?
        # while len(recent_moves) > ntabu:
        while seg.segment_length() > ntabu:
            move = recent_moves.pop(0)
            seg = move.apply_move_left_end(seg)
        move_iter = iter(nbhood.random_moves_right_end_farther_without_replacement(seg))
        move = next(move_iter, None)
    # FIXME: should return the best solution found, not the last
    return best
