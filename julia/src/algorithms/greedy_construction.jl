"""
    greedy_construction(problem; solution=nothing)

Solve `problem` using a greedy construction approach.
If `solution` is given, it is used as the starting point.
"""
function greedy_construction(problem; solution=nothing)
    neigh = construction_neighbourhood(problem)

    if solution === nothing
        solution = empty_solution(problem)
    end

    mvs = _gc_moves_and_increments(neigh, solution)
    while !isempty(mvs)
        best = argmin(m -> m[2], mvs)
        best_move = best[1]
        solution = apply_move(best_move, solution)

        mvs = _gc_moves_and_increments(neigh, solution)
    end

    return solution
end

"""
    greedy_construction_with_random_tie_breaking(problem; solution=nothing)

Solve `problem` using a greedy construction approach with random tie-breaking.
"""
function greedy_construction_with_random_tie_breaking(problem; solution=nothing)
    neigh = construction_neighbourhood(problem)

    if solution === nothing
        solution = empty_solution(problem)
    end

    mvs = _gc_moves_and_increments(neigh, solution)
    while !isempty(mvs)
        best_incr = minimum(m -> m[2], mvs)
        best_moves = [m[1] for m in mvs if m[2] <= best_incr + 1e-6]
        solution = apply_move(rand(best_moves), solution)

        mvs = _gc_moves_and_increments(neigh, solution)
    end

    return solution
end

function _gc_moves_and_increments(neigh, solution)
    result = []
    for mv in moves(neigh, solution)
        incr = lower_bound_increment(mv, solution)
        if incr !== nothing
            push!(result, (mv, incr))
        end
    end
    return result
end
