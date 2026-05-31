"""
    best_improvement(problem, solution)

Improve `solution` by repeatedly applying the best improving move
from the local neighbourhood, until no improving move exists.
"""
function best_improvement(problem, solution)
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

function _bi_moves_and_increments(neigh, solution)
    result = []
    for mv in moves(neigh, solution)
        incr = objective_value_increment(mv, solution)
        @assert incr !== nothing
        if incr < 0
            push!(result, (mv, incr))
        end
    end
    return result
end
