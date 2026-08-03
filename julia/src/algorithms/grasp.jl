"""
    grasp(problem::Problem, budget::Real; solution::Union{Nothing, Solution}=nothing,
          alpha::Real=0.1, local_search::Union{Nothing, Function}=nothing) -> solution::Solution

Greedy Randomized Adaptive Search Procedure: build solutions using a
restricted candidate list controlled by `alpha`, then optionally improve
with `local_search`. Runs for `budget` seconds.
"""
function grasp(problem::Problem, budget::Real;
        solution::Union{Nothing, Solution}=nothing,
        alpha::Real=0.1,
        local_search::Union{Nothing, Function}=nothing)::Solution

    start = time()

    neigh = construction_neighbourhood(problem)

    if solution === nothing
        solution = empty_solution(problem)
    end

    best = solution
    best_obj = objective_value(solution)

    while time() - start < budget
        s = copy_solution(solution)
        b::Union{Nothing, Solution} = nothing
        b_obj::Union{Nothing, <:Real} = nothing

        cl = _gr_moves_and_increments(neigh, s)
        while !isempty(cl)
            cmin = minimum(m -> m[2], cl)
            cmax = maximum(m -> m[2], cl)
            thresh = cmin + alpha * (cmax - cmin)
            rcl = [m[1] for m in cl if m[2] <= thresh]
            mv = rand(rcl)
            s = apply_move(mv, s)
            obj = objective_value(s)
            if obj !== nothing && (b_obj === nothing || obj < b_obj)
                b = copy_solution(s)
                b_obj = objective_value(b)
            end
            cl = _gr_moves_and_increments(neigh, s)
        end

        if b !== nothing
            if local_search !== nothing
                b = local_search(problem, b)
                b_obj = objective_value(b)
            end
            if best_obj === nothing || b_obj < best_obj
                @info "Best solution: $b_obj"
                best = b
                best_obj = b_obj
            end
        end
    end

    return best
end

function _gr_moves_and_increments(neigh::Neighbourhood, solution::Solution)::Vector{Tuple{Any, <:Real}}
    result::Vector{Tuple{Any, <:Real}} = Tuple{Any, <:Real}[]
    for mv in moves(neigh, solution)
        incr = lower_bound_increment(mv, solution)
        if incr !== nothing
            push!(result, (mv, incr))
        end
    end
    return result
end
