
# SPDX-FileCopyrightText: © 2025 Authors of the roar-net-api-py project <https://github.com/roar-net/roar-net-api-py/blob/main/AUTHORS>
#
# SPDX-License-Identifier: Apache-2.0
"""
    objective_value(solution::Solution) -> value::Union{Nothing, <:Real}

Return the objective value of `solution`, or `nothing` if the solution
is not feasible.
"""
function objective_value(::Solution)::Union{Nothing, <:Real} end
