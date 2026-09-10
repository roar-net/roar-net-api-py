
# SPDX-FileCopyrightText: © 2025 Authors of the roar-net-api-py project <https://github.com/roar-net/roar-net-api-py/blob/main/AUTHORS>
#
# SPDX-License-Identifier: Apache-2.0
"""
    random_move(neighbourhood::Neighbourhood, solution::Solution) -> move::Union{Nothing, Move}

Return a single random `Move` from `neighbourhood` for `solution`,
or `nothing` if no move exists.
"""
function random_move(::Neighbourhood, ::Solution)::Union{Nothing, Move} end
