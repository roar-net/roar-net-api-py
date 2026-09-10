# SPDX-FileCopyrightText: © 2025 Authors of the roar-net-api-py project <https://github.com/roar-net/roar-net-api-py/blob/main/AUTHORS>
#
# SPDX-License-Identifier: Apache-2.0

"""
    best_improvement(problem::Problem, solution::Solution) -> solution::Solution

Improve `solution` by repeatedly applying the best improving move
from the local neighbourhood, until no improving move exists.
"""
function best_improvement(problem::Problem, solution::Solution)::Solution
    neigh = local_neighbourhood(problem)

    mvs = _bi_moves_and_increments(neigh, solution)
    while !isempty(mvs)
        best = argmin(m -> m[2], mvs)
        best_move = best[1]
        best_incr = best[2]

        @info "Best increment: $best_incr"

        solution = apply_move(best_move, solution)
        mvs = _bi_moves_and_increments(neigh, solution)
    end

    return solution
end

function _bi_moves_and_increments(neigh::Neighbourhood, solution::Solution)::Vector{Tuple{Any, <:Real}}
    result::Vector{Tuple{Any, <:Real}} = Tuple{Any, <:Real}[]
    for mv in moves(neigh, solution)
        incr = objective_value_increment(mv, solution)
        @assert incr !== nothing
        if incr < 0
            push!(result, (mv, incr))
        end
    end
    return result
end
