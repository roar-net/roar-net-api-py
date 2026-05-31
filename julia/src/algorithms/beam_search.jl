"""
    KMin{K, V}

Data structure that maintains the `k` smallest items according to a key function.
"""
mutable struct KMin{K,V}
    k::Int
    key::Function
    keys::Vector{K}
    values::Vector{V}
end

function KMin(k::Int, key::Function)
    return KMin(k, key, Vector{Any}(), Vector{Any}())
end

function Base.push!(km::KMin, value)
    key = km.key(value)
    if length(km.values) == km.k && key > km.keys[end]
        return
    end
    i = searchsortedlast(km.keys, key) + 1
    insert!(km.keys, i, key)
    insert!(km.values, i, value)
    if length(km.values) > km.k
        pop!(km.keys)
        pop!(km.values)
    end
end

Base.length(km::KMin) = length(km.values)
Base.iterate(km::KMin) = iterate(km.values)
Base.iterate(km::KMin, state) = iterate(km.values, state)

"""
    beam_search(problem; solution=nothing, bw=10)

Beam search: a constructive heuristic that maintains a beam of width `bw`
partial solutions, extending each by the best moves at each step.
"""
function beam_search(problem; solution=nothing, bw=10)
    neigh = construction_neighbourhood(problem)

    if solution === nothing
        solution = empty_solution(problem)
    end

    best = solution
    best_obj = objective_value(best)

    lb = lower_bound(best)
    if lb === nothing
        return best
    end

    v = [(lb, best)]

    while true
        candidates = KMin(bw, x -> x[1])
        for (lb_val, s) in v
            for mv in moves(neigh, s)
                incr = lower_bound_increment(mv, s)
                if incr !== nothing
                    push!(candidates, (lb_val + incr, s, mv))
                end
            end
        end

        if length(candidates) == 0
            break
        end

        v = []
        for (lb_val, s, mv) in candidates
            ns = copy_solution(s)
            ns = apply_move(mv, ns)
            push!(v, (lb_val, ns))
            obj = objective_value(ns)
            if obj !== nothing && (best_obj === nothing || obj < best_obj)
                @info "Best solution: $obj"
                best = ns
                best_obj = obj
            end
        end
    end

    return best
end
