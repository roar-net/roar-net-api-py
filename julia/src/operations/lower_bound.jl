# SPDX-FileCopyrightText: © 2025 Authors of the roar-net-api-py project <https://github.com/roar-net/roar-net-api-py/blob/main/AUTHORS>
#
# SPDX-License-Identifier: Apache-2.0
"""
    lower_bound(solution::Solution) -> value::Union{Nothing, <:Real}

Return a lower bound on the objective value of `solution`.
Used by beam search and other constructive algorithms.
"""
function lower_bound(::Solution)::Union{Nothing, <:Real} end
