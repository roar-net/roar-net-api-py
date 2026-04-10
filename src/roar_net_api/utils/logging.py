# SPDX-FileCopyrightText: © 2025 Authors of the roar-net-api-py project <https://github.com/roar-net/roar-net-api-py/blob/main/AUTHORS>
#
# SPDX-License-Identifier: Apache-2.0

import csv
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from functools import wraps
from logging import getLogger
from time import perf_counter_ns
from typing import Generator, TextIO, Final, TypeVar, Any, ParamSpec

log = getLogger(__name__)

T = TypeVar("T")
P = ParamSpec("P")


class PerformanceLogger:
    RESERVED_ATTRIBUTES: Final[set[str]] = {"run_id"}

    def __init__(self) -> None:
        self._data: list[tuple[object, ...]] = []
        self._attributes: dict[str, str] = {}
        self._run_id: int = 0
        self._run_data: list[tuple[int, str]] = []
        self._run_active: bool = False
        self._run_start: int = 0

        self._problem_operations_to_wrap: Mapping[str, Callable[..., Any]] = {
            "empty_solution": self._wrap_solution_creation_method,
            "random_solution": self._wrap_solution_creation_method,
        }

        self._solution_operations_to_wrap: Mapping[str, Callable[..., Any]] = {
            "copy_solution": self._wrap_solution_creation_method,
            "objective_value": self._wrap_objective_value,
        }

    def set_attribute(self, key: str, value: str) -> None:
        """
        Sets an attribute. If the attribute already existed it is
        silently updated.
        """
        if key in self.RESERVED_ATTRIBUTES:
            log.warning("%s is a reserved attribute and the set value will be ignored" % key)
            return

        self._attributes[key] = value

    def problem(self, problem: T) -> T:
        return self._wrap_problem_operations(problem)

    @contextmanager
    def run(self) -> Generator[None, None, None]:
        """ """
        self._run_id += 1
        self._run_active = True
        self._run_start = perf_counter_ns()
        self._run_data = []

        yield

        self._data.extend((self._run_id, t, f, *self._attributes.values()) for t, f in self._run_data)

        self._run_active = False

    def write(self, textio: TextIO) -> None:
        # IMPROVE: We could add different formats here with a "format" argument
        fieldnames = ["index", "time", "fval", *self._attributes.keys()]
        writer = csv.DictWriter(textio, fieldnames=fieldnames)
        writer.writeheader()
        for record in self._data:
            row = dict(zip(fieldnames, record))
            writer.writerow(row)

    def _wrap_problem_operations(self, problem: T) -> T:
        for op, wrapper in self._problem_operations_to_wrap.items():
            if hasattr(problem, op):
                # log.info("Decorating problem operation %s" % op)
                setattr(problem, op, wrapper(getattr(problem, op)))
        return problem

    def _wrap_solution_operations(self, solution: T) -> T:
        for op, wrapper in self._solution_operations_to_wrap.items():
            if hasattr(solution, op):
                # log.info("Decorating solution operation %s" % op)
                setattr(solution, op, wrapper(getattr(solution, op)))
        return solution

    def _wrap_solution_creation_method(self, f: Callable[P, T]) -> Callable[P, T]:
        @wraps(f)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            solution = f(*args, **kwargs)
            return self._wrap_solution_operations(solution)

        return wrapper

    def _wrap_objective_value(self, f: Callable[P, T]) -> Callable[P, T]:
        @wraps(f)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            obj = f(*args, **kwargs)
            self._log_objective_value(obj)
            return obj

        return wrapper

    def _log_objective_value(self, val: Any) -> None:
        if not self._run_active:
            log.debug("Run not active, ignoring objective value log")
            return

        self._run_data.append((perf_counter_ns() - self._run_start, str(val)))
