# SPDX-FileCopyrightText: © 2025 Authors of the roar-net-api-py project <https://github.com/roar-net/roar-net-api-py/blob/main/AUTHORS>
#
# SPDX-License-Identifier: Apache-2.0
"""
    lower_bound_increment(move::Move, solution::Solution) -> value::Union{Nothing, <:Real}

Return the lower bound increment for applying `move` to `solution`,
or `nothing` if the move is not applicable. Used by constructive algorithms.
"""
function lower_bound_increment(::Move, ::Solution)::Union{Nothing, <:Real} end
