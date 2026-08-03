"""
    first_improvement(problem::Problem, solution::Solution) -> solution::Solution

Improve `solution` by scanning the local neighbourhood in random order
and applying the first improving move found.
"""
function first_improvement(problem::Problem, solution::Solution)::Solution
    neigh = local_neighbourhood(problem)

    while true
        found = false
        for mv in random_moves_without_replacement(neigh, solution)
            incr = objective_value_increment(mv, solution)
            @assert incr !== nothing
            if incr < 0
                @info "Found increment: $incr"
                solution = apply_move(mv, solution)
                found = true
                break
            end
        end
        if !found
            break
        end
    end

    return solution
end
