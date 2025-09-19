# SPDX-FileCopyrightText: © 2025 Authors of the roar-net-api-py project <https://github.com/roar-net/roar-net-api-py/blob/main/AUTHORS>
#
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Any, Callable, Optional, Union, Type
import csv

from roar_net_api.types import (
    Problem,
    Solution,
)

perflog = logging.getLogger("PerformanceLogger")


class ListLogger(logging.Handler):
    def __init__(self, level: int = 5):
        super().__init__(level=level)
        self.records: list[str] = []
        self.level = level

    def emit(self, record: logging.LogRecord) -> None:
        if record.levelno == self.level:
            self.records.append(self.format(record))


def logged(func: Callable) -> Callable:
    def wrapper(*args, **kwargs) -> Optional[int]:
        result = func(*args, **kwargs)
        if result is not None:
            logging.getLogger("PerformanceLogger").log(5, f"{result}")
        return result

    return wrapper


class PerformanceLogger:
    def __init__(self, filename: Optional[str] = None, algname: Optional[str] = None):
        self.run_id: int = 0
        self.finished_runs: list[tuple[Union[int, float, str]]] = []
        self.filename = filename if filename is not None else "performance_log.csv"
        self.algname = algname
        self.logger = ListLogger()
        self.logger.setFormatter(logging.Formatter("%(created)f %(message)s"))
        global perflog
        perflog.addHandler(self.logger)
        perflog.setLevel(5)

    def reset(self) -> None:
        if self.logger.records is not None and len(self.logger.records) > 1:
            self.finished_runs += self.process_run()
            self.logger.records.clear()
        perflog.log(level=5, msg="inf")
        self.run_id += 1
        return

    def add_attribute(self, key: str, value: str) -> None:
        if self.logger.records is not None and len(self.logger.records) > 1:
            self.reset()
        if not hasattr(self, "attributes"):
            self.attributes = {}
        self.attributes[key] = value

    def process_run(self) -> list[tuple[Union[int, float, str]]]:
        times_tuple, fvals = zip(*[entry.split(" ", 1) for entry in self.logger.records])
        times = [float(t) - float(times_tuple[0]) for t in times_tuple]
        attributes = getattr(self, "attributes", {})
        records = []
        for t, f in zip(times, fvals):
            record = tuple([int(self.run_id), int(t * 1e6) + 1, float(f), *attributes.values()])
            records.append(record)
        return records

    def save_runs(self) -> list[tuple[Union[int, float, str]]]:
        fieldnames = ["index", "time", "fval", *(getattr(self, "attributes", {}).keys())]
        with open(self.filename, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for record in self.finished_runs:
                row = dict(zip(fieldnames, record))
                writer.writerow(row)

        return self.finished_runs

    def close(self) -> list[tuple[Union[int, float, str]]]:
        self.reset()
        global perflog
        if self.logger:
            perflog.removeHandler(self.logger)
        return self.save_runs()
