
# SPDX-FileCopyrightText: © 2025 Authors of the roar-net-api-py project <https://github.com/roar-net/roar-net-api-py/blob/main/AUTHORS>
#
# SPDX-License-Identifier: Apache-2.0
"""
    rls(problem::Problem, solution::Solution, budget::Real) -> solution::Solution

Random Local Search: repeatedly sample random moves from the local
neighbourhood and apply any non-worsening move, until `budget` seconds
have elapsed or no improving move is found.
"""
function rls(problem::Problem, solution::Solution, budget::Real)::Solution
    start = time()

    neigh = local_neighbourhood(problem)

    while time() - start < budget
        found = false
        for mv in random_moves_without_replacement(neigh, solution)
            incr = objective_value_increment(mv, solution)
            @assert incr !== nothing
            if incr <= 0
                @info "Found increment: $incr"
                solution = apply_move(mv, solution)
                found = true
                break
            end
            if time() - start >= budget
                return solution
            end
        end
        if !found
            break
        end
    end

    return solution
end
