#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: © 2025 Authors of the roar-net-api-py project <https://github.com/roar-net/roar-net-api-py/blob/main/AUTHORS>
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable
from typing import Optional, Protocol, TypeVar


class SupportsSegmentLength(Protocol):
    def segment_length(self) -> int: ...


_TSegment = TypeVar("_TSegment")


class SupportsApplyMoveLeftEnd(Protocol[_TSegment]):
    def apply_move_left_end(self, segment: _TSegment) -> _TSegment: ...


class SupportsApplyMoveRightEnd(Protocol[_TSegment]):
    def apply_move_right_end(self, segment: _TSegment) -> _TSegment: ...


_TSegment_co = TypeVar("_TSegment_co", covariant=True)
_TSolution_contra = TypeVar("_TSolution_contra", contravariant=True)


class SupportsSegment(Protocol[_TSolution_contra, _TSegment_co]):
    def segment(self, solution0: _TSolution_contra, solution1: _TSolution_contra) -> Optional[_TSegment_co]: ...


_TSegment_contra = TypeVar("_TSegment_contra", contravariant=True)
_TMove_co = TypeVar("_TMove_co", covariant=True)


class SupportsRandomMovesRightEndFartherWithoutReplacement(Protocol[_TSegment_contra, _TMove_co]):
    def random_moves_right_end_farther_without_replacement(self, segment: _TSegment_contra) -> Iterable[_TMove_co]: ...
