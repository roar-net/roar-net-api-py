# SPDX-FileCopyrightText: © 2025 Authors of the roar-net-api-py project <https://github.com/roar-net/roar-net-api-py/blob/main/AUTHORS>
#
# SPDX-License-Identifier: Apache-2.0
"""
    moves(neighbourhood::Neighbourhood, solution::Solution) -> iterable::Any

Return an iterable of all valid `Move`s in `neighbourhood` for `solution`.
The return type is declared `Any` because concrete implementations may use
either a `Vector{<:Move}` or a `Channel{<:Move}`.
"""
function moves(::Neighbourhood, ::Solution)::Any end
