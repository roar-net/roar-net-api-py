# SPDX-FileCopyrightText: © 2025 Authors of the roar-net-api-py project <https://github.com/roar-net/roar-net-api-py/blob/main/AUTHORS>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Protocol, Union


class SupportsObjectiveValue(Protocol):
    def objective_value(self) -> Optional[Union[int, float]]: ...
