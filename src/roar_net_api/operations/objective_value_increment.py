# SPDX-FileCopyrightText: © 2025 Authors of the roar-net-api-py project <https://github.com/roar-net/roar-net-api-py/blob/main/AUTHORS>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Protocol, TypeVar, Union

Solution = TypeVar("Solution", contravariant=True)


class SupportsObjectiveValueIncrement(Protocol[Solution]):
    def objective_value_increment(self, solution: Solution) -> Optional[Union[int, float]]: ...
