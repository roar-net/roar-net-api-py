# SPDX-FileCopyrightText: © 2025 Authors of the roar-net-api-py project <https://github.com/roar-net/roar-net-api-py/blob/main/AUTHORS>
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable
from typing import Protocol, TypeVar

_TSubNeighbourhoods_co = TypeVar("_TSubNeighbourhoods_co", covariant=True, bound=Iterable[object])


class SupportsSubNeighbourhoods(Protocol[_TSubNeighbourhoods_co]):
    def sub_neighbourhoods(self) -> _TSubNeighbourhoods_co: ...
