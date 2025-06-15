# SPDX-FileCopyrightText: © 2025 Authors of the roar-net-api-py project <https://github.com/roar-net/roar-net-api-py/blob/main/AUTHORS>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Protocol, TypeVar

_TSolution_contra = TypeVar("_TSolution_contra", contravariant=True)
_TIncrement_co = TypeVar("_TIncrement_co", covariant=True)


class SupportsObjectiveValueIncrement(Protocol[_TSolution_contra, _TIncrement_co]):
    def objective_value_increment(self, solution: _TSolution_contra) -> Optional[_TIncrement_co]: ...
