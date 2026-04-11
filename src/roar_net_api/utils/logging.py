# SPDX-FileCopyrightText: © 2025 Authors of the roar-net-api-py project <https://github.com/roar-net/roar-net-api-py/blob/main/AUTHORS>
#
# SPDX-License-Identifier: Apache-2.0

import csv
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from logging import getLogger
from time import perf_counter_ns
from typing import TextIO, Final, Any, TypeVar, Type, cast

log = getLogger(__name__)

T = TypeVar("T")


class Singleton(type):
    _instances: dict[type, Any] = {}

    def __call__(cls: Type[T], *args: Any, **kwargs: Any) -> T:
        if cls not in Singleton._instances:
            Singleton._instances[cls] = super(Singleton, cast(Singleton, cls)).__call__(*args, **kwargs)
        return cast(T, Singleton._instances[cls])


@dataclass(frozen=True)
class _Record:
    run_id: int
    time_ns: int
    obj_val: int | float | None
    algorithm: str
    attributes: dict[str, str]


class PerformanceLogger(metaclass=Singleton):
    RESERVED_ATTRIBUTES: Final[set[str]] = {"run_id", "time_ns", "obj_val", "algorithm"}

    def __init__(self) -> None:
        self._data: list[_Record] = []
        self._attributes: dict[str, str] = {}
        self._run_id: int = 0
        self._run_data: list[_Record] = []
        self._run_active: bool = False
        self._run_start: int = 0

    def set_attribute(self, key: str, value: str) -> None:
        """
        Sets an attribute. If the attribute already existed it is
        silently updated.
        """
        if key in self.RESERVED_ATTRIBUTES:
            log.warning("%s is a reserved attribute and the set value will be ignored" % key)
            return

        self._attributes[key] = value

    @contextmanager
    def run(self) -> Generator[None, None, None]:
        if self._run_active:
            log.warning("run already active, ignoring new run")
            yield
            return

        self._run_id += 1
        self._run_active = True
        self._run_start = perf_counter_ns()
        self._run_data = []

        yield

        self._data.extend(self._run_data)
        self._run_data = []
        self._run_active = False

    def write(self, textio: TextIO) -> None:
        # IMPROVE: We could add different formats here with a "format" argument

        # Since we can't remove attributes, the last _attributes value
        # will contain all possible keys
        keys = self._attributes.keys()
        fieldnames = ["run_id", "time_ns", "fval", "algorithm", *keys]
        writer = csv.DictWriter(textio, fieldnames=fieldnames)
        writer.writeheader()
        for r in self._data:
            row = {
                "run_id": r.run_id,
                "time_ns": r.time_ns,
                "fval": r.obj_val,
                "algorithm": r.algorithm,
                **r.attributes,
            }
            writer.writerow(row)

    def log(self, val: int | float | None, algorithm: str) -> None:
        if not self._run_active:
            log.debug("Run not active, ignoring objective value log")
            return

        self._run_data.append(
            _Record(
                run_id=self._run_id,
                time_ns=perf_counter_ns() - self._run_start,
                obj_val=val,
                algorithm=algorithm,
                attributes=self._attributes.copy(),
            )
        )

    @property
    def active(self) -> bool:
        return self._run_active
