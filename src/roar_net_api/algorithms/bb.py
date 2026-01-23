# SPDX-FileCopyrightText: © 2025 Authors of the roar-net-api-py project <https://github.com/roar-net/roar-net-api-py/blob/main/AUTHORS>
#
# SPDX-License-Identifier: Apache-2.0

from logging import getLogger
from typing import Protocol, TypeVar, Optional

from ..operations import (
    SupportsApplyMove,
    SupportsCopySolution,
    SupportsConstructionNeighbourhood,
    SupportsEmptySolution,
    SupportsLowerBoundIncrement,
    SupportsLowerBound,
    SupportsObjectiveValue,
    SupportsMoves,
    SupportsRevertMove,
)

log = getLogger(__name__)


class _Solution(SupportsLowerBound, SupportsObjectiveValue, SupportsCopySolution, Protocol): ...


_TSolution = TypeVar("_TSolution", bound=_Solution)


class _Move(
    SupportsApplyMove[_TSolution],
    SupportsRevertMove[_TSolution],
    SupportsLowerBoundIncrement[_TSolution],
    Protocol,
): ...


class _Neighbourhood(SupportsMoves[_TSolution, _Move[_TSolution]], Protocol): ...


class _Problem(
    SupportsEmptySolution[_TSolution], SupportsConstructionNeighbourhood[_Neighbourhood[_TSolution]], Protocol
): ...


def bb(problem: _Problem[_TSolution], solution: Optional[_TSolution] = None) -> _TSolution:
    neigh = problem.construction_neighbourhood()

    if solution is None:
        solution = problem.empty_solution()

    # Stack keeps the list of moves for the solution at that level.
    # Since we cannot assume the iterator remains valid after applying
    # and reverting the move, we must keep the list of moves in the
    # stack.
    stack = [list(iter(neigh.moves(solution)))]

    # List of applied moves at each level, such that we can revert
    # them when going back "up" in the search tree.
    moves: list[_Move[_TSolution]] = []

    lb = solution.lower_bound()
    if lb is None:
        # If the current solution has no lower bound, it means no feasible
        # solution can be achieved by construction, so we can return.
        return solution

    obj = solution.objective_value()

    # List of lower bounds (to allow incremental computation)
    lbs = [lb]

    # Best objective (to cut solution that are provably non-optimal)
    best_obj = obj
    best_solution = None

    while stack:
        if len(stack[-1]) == 0:
            if len(moves) > 0:
                solution = moves.pop().revert_move(solution)
            lbs.pop()
            stack.pop()
            continue

        move = stack[-1].pop()
        lb_incr = move.lower_bound_increment(solution)
        if lb_incr is None:
            # No feasible solutions exist in the branch, so we can skip it
            continue

        new_lb = lbs[-1] + lb_incr

        if best_obj is not None and new_lb >= best_obj:
            # We can skip this branch, since it has no solution better
            # than our current best
            continue

        solution = move.apply_move(solution)

        new_obj = solution.objective_value()
        if new_obj is not None and (best_obj is None or new_obj < best_obj):
            best_obj = new_obj
            best_solution = solution.copy_solution()

        stack.append(list(iter(neigh.moves(solution))))
        moves.append(move)
        lbs.append(new_lb)

    return best_solution if best_solution is not None else solution
