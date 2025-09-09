#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: © 2025 Authors of the roar-net-api-py project <https://github.com/roar-net/roar-net-api-py/blob/main/AUTHORS>
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import random
import sys
import json
from collections.abc import Iterable, Sequence
from logging import getLogger
from typing import Optional, Protocol, Self, TextIO, TypeVar, final

from roar_net_api.operations import (
    SupportsApplyMove,
    SupportsConstructionNeighbourhood,
    SupportsCopySolution,
    SupportsEmptySolution,
    SupportsLocalNeighbourhood,
    SupportsLowerBound,
    SupportsLowerBoundIncrement,
    SupportsMoves,
    SupportsObjectiveValue,
    SupportsObjectiveValueIncrement,
    SupportsRandomMove,
    SupportsRandomMovesWithoutReplacement,
    SupportsRandomSolution,
)

log = getLogger(__name__)


class _SupportsLT(Protocol):
    def __lt__(self, other: Self) -> bool: ...


_T = TypeVar("_T", bound=_SupportsLT)


class SessionOccurrence:
    def __init__(self, name: str):
        self.name = name

    def __repr__(self):
        return self.name


class TeachingAssistant:
    def __init__(self, name: str, qualifications: dict[SessionOccurrence, int]):
        self.name = name
        self.qualifications = qualifications

    def __repr__(self):
        # return "Teaching Assistant: ".join(self.name) #+ ": qualifications: ".join(self.qualifications)
        # TODO: add qualifications to the method as well
        return self.name


# ---------------------------------- Solution --------------------------------


@final
class Solution(SupportsCopySolution, SupportsObjectiveValue, SupportsLowerBound):
    def __init__(self, problem: Problem, mapping: dict[SessionOccurrence, TeachingAssistant],
                 unused_tas: list[TeachingAssistant], unused_session_occurrences: list[SessionOccurrence], lb: int):
        self.problem = problem
        self.mapping = mapping
        self.unused_tas = unused_tas
        self.unused_session_occurrences = unused_session_occurrences
        self.lb = lb

    def __str__(self) -> str:
        # TODO: Maybe add the unused objects here
        return " ".join(map(str, self.mapping))

    @property
    def is_feasible(self) -> bool:
        return len(self.unused_session_occurrences) == 0

    def to_textio(self, f: TextIO) -> None:
        f.write("NAME : %s\nTYPE : TA\n" % self.problem.name)
        f.write("#SessionOccurrences : %d\n" % len(self.problem.session_occurrences))
        f.write("#TeachingAssistants : %d\n" % len(self.problem.tas))
        for so in self.mapping.keys():
            f.write(so.name + " -> " + self.mapping[so].name + "\n")
        f.write("EOF\n")

    def copy_solution(self) -> Self:
        return self.__class__(self.problem, self.mapping.copy(), self.unused_tas.copy(),
                              self.unused_session_occurrences.copy(), self.lb)

    def objective_value(self) -> Optional[int]:
        if self.is_feasible:
            return self.lb
        return None

    def lower_bound(self) -> int:
        return self.lb


# ----------------------------------- Moves -----------------------------------


@final
class AddMove(SupportsApplyMove[Solution], SupportsLowerBoundIncrement[Solution]):
    def __init__(self, neighbourhood: AddNeighbourhood, session_occurrence: SessionOccurrence, ta: TeachingAssistant):
        self.neighbourhood = neighbourhood
        self.session_occurrence = session_occurrence
        self.ta = ta

    def apply_move(self, solution: Solution) -> Solution:
        prob = solution.problem
        # Update lower bound: += qualification of the newly added TA mapping
        solution.lb += self.ta.qualifications.get(self.session_occurrence)
        # Update solution
        solution.mapping[self.session_occurrence] = self.ta
        solution.unused_session_occurrences.remove(self.session_occurrence)
        if self.ta in solution.unused_tas:
            solution.unused_tas.remove(self.ta)
        return solution

    def lower_bound_increment(self, solution: Solution) -> float:
        return self.ta.qualifications.get(self.session_occurrence)


# TODO: Rename to swap
@final
class NewTaMove(SupportsApplyMove[Solution], SupportsObjectiveValueIncrement[Solution]):
    def __init__(self, neighbourhood: NewTaNeighbourhood, so: SessionOccurrence, new_ta: TeachingAssistant):
        # The new TA must have at least some qualification for the session occurrence
        assert new_ta.qualifications[so] is not None
        self.neighbourhood = neighbourhood
        self.so = so
        self.ta = new_ta

    def apply_move(self, solution: Solution) -> Solution:
        # Update/decrement the lb by removing the previous mappings
        solution.lb -= solution.mapping[self.so].qualifications.get(self.so)

        # Swap TAs
        solution.mapping[self.so] = self.ta

        # Update lb with the new mappings
        solution.lb += solution.mapping[self.so].qualifications.get(self.so)
        return solution

    def objective_value_increment(self, solution: Solution) -> float:
        incr = 0
        incr -= solution.mapping[self.so].qualifications.get(self.so)
        incr += self.ta.qualifications.get(self.so)
        return incr


# ------------------------------- Neighbourhood ------------------------------


@final
class AddNeighbourhood(SupportsMoves[Solution, AddMove]):
    def __init__(self, problem: Problem):
        self.problem = problem

    def moves(self, solution: Solution) -> Iterable[AddMove]:
        assert self.problem == solution.problem
        # TODO: This statement always uses the last unused TA. I'm not sure if this makes sense in every case.
        for so in solution.unused_session_occurrences:
            ta = solution.unused_tas[-1]
            assert so in ta.qualifications
            yield AddMove(self, so, ta)


# TODO: Rename to swap
@final
class NewTaNeighbourhood(
    SupportsMoves[Solution, NewTaMove],
    SupportsRandomMovesWithoutReplacement[Solution, NewTaMove],
    SupportsRandomMove[Solution, NewTaMove],
):
    def __init__(self, problem: Problem):
        self.problem = problem

    def moves(self, solution: Solution) -> Iterable[NewTaMove]:
        assert self.problem == solution.problem
        # This is only meant to be used as a local neighbourhood, so solution should be feasible
        assert solution.is_feasible

        # All TAs can be assigned except if they are the same
        for so in self.problem.session_occurrences:
            for ta in self.problem.tas:
                # The TA must not be the same
                if solution.mapping[so] != ta:
                    # The TA must have some qualifications for the session occurrence
                    if so in ta.qualifications:
                        yield NewTaMove(self, so, ta)

    def random_moves_without_replacement(self, solution: Solution) -> Iterable[NewTaMove]:
        assert self.problem == solution.problem
        sos = solution.problem.session_occurrences.copy()
        random.shuffle(sos)
        for idx, x in enumerate(sos):
            tas = solution.problem.tas.copy()
            random.shuffle(tas)
            for idy, y in enumerate(tas):
                # The TA must not be the same
                if solution.mapping[x] != y:
                    # The TA must have some qualifications for the session occurrence
                    if x in y.qualifications:
                        yield NewTaMove(self, x, y)

    def random_move(self, solution: Solution) -> Optional[NewTaMove]:
        return next(iter(self.random_moves_without_replacement(solution)), None)


# ---------------------------------- Problem --------------------------------


@final
class Problem(
    SupportsConstructionNeighbourhood[AddNeighbourhood],
    SupportsLocalNeighbourhood[NewTaNeighbourhood],
    SupportsEmptySolution[Solution],
    SupportsRandomSolution[Solution],
):
    def __init__(self, session_occurrences: list[SessionOccurrence], tas: list[TeachingAssistant], name: str):
        self.session_occurrences = session_occurrences
        self.tas = tas
        self.name = name
        self.n = len(self.session_occurrences)
        self.c_nbhood: Optional[AddNeighbourhood] = None
        self.l_nbhood: Optional[NewTaNeighbourhood] = None

    def __str__(self) -> str:
        out: list[str] = []
        out.append("Session occurrences:\n")
        for so in self.session_occurrences:
            out.append(" ".join(so.name))
        out.append("Teaching assistants:\n")
        for ta in self.tas:
            out.append(" ".join(ta.name))
        return "\n".join(out)

    def construction_neighbourhood(self) -> AddNeighbourhood:
        if self.c_nbhood is None:
            self.c_nbhood = AddNeighbourhood(self)
        return self.c_nbhood

    def local_neighbourhood(self) -> NewTaNeighbourhood:
        if self.l_nbhood is None:
            self.l_nbhood = NewTaNeighbourhood(self)
        return self.l_nbhood

    @classmethod
    def from_json(cls, path: str) -> Self:
        with open(path) as f:
            imported = json.load(f)

            # Session occurrences
            imported_session_occurrences = {}
            for so in imported["sessionOccurrences"]:
                imported_session_occurrences[so] = SessionOccurrence(so)

            # TAs
            imported_tas = []
            for ta in imported["tas"]:
                leftover_session_occurrences = imported_session_occurrences.copy()
                qualifications = {}
                for q in ta["qualifications"].keys():
                    # The `qualification` value must be negative because the ROAR-NET API only minimizes the obj value
                    qualifications[imported_session_occurrences[q]] = -1 * ta["qualifications"].get(q)
                    leftover_session_occurrences.pop(q)
                # Fill all other qualifications up with '0'
                for q in leftover_session_occurrences.values():
                    qualifications[q] = 0
                new_ta = TeachingAssistant(ta["name"], qualifications)
                imported_tas.append(new_ta)

            return cls(list(imported_session_occurrences.values()), imported_tas, imported["name"])

    def empty_solution(self) -> Solution:
        return Solution(self, {}, self.tas.copy(), self.session_occurrences.copy(), 0)

    def random_solution(self) -> Solution:
        mapping = {}
        obj = 0
        unused_tas = self.tas.copy()
        for so in self.session_occurrences:
            mapping[so] = random.choice(self.tas)
            obj += mapping[so].qualifications[so]
            unused_tas.remove(mapping[so])
        return Solution(self, mapping.copy(), unused_tas.copy(), [], obj)


if __name__ == "__main__":
    import roar_net_api.algorithms as alg

    logging.basicConfig(stream=sys.stderr, level="INFO", format="%(levelname)s;%(asctime)s;%(message)s")

    problem = Problem.from_json('./ta.json')

    # exit(0)

    # problem = Problem.from_textio(sys.stdin)

    # Run greedy construction to get an initial solution
    solution = alg.greedy_construction(problem)
    # solution = alg.beam_search(problem, bw=10)
    # solution = alg.grasp(problem, 30.0)
    log.info(f"Objective value after constructive search: {solution.objective_value()}")

    log.info("Initial solution:")
    solution.to_textio(sys.stdout)

    # Run simulated annealing to improve the previous solution
    solution = alg.sa(problem, solution, 2.0, 30.0)
    # solution = alg.rls(problem, solution, 10.0)
    # solution = alg.best_improvement(problem, solution)
    # solution = alg.first_improvement(problem, solution)
    log.info(f"Objective value after local search: {solution.objective_value()}")

    # Print the final solution to stdout
    solution.to_textio(sys.stdout)
