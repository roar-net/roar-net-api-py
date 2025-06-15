# SPDX-FileCopyrightText: © 2025 Authors of the roar-net-api-py project <https://github.com/roar-net/roar-net-api-py/blob/main/AUTHORS>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Protocol, TypeVar


_TValue_co = TypeVar("_TValue_co", covariant=True)


class SupportsLowerBound(Protocol[_TValue_co]):
    def lower_bound(self) -> Optional[_TValue_co]: ...
