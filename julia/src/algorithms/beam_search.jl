# SPDX-FileCopyrightText: © 2025 Authors of the roar-net-api-py project <https://github.com/roar-net/roar-net-api-py/blob/main/AUTHORS>
#
# SPDX-License-Identifier: Apache-2.0

"""
    KMin{K, V}

Data structure that maintains the `k` smallest items according to a key function.
The element types `K` and `V` are concrete per call; the default constructor
returns `KMin{Any, Any}` so callers can push items of any type.
"""
mutable struct KMin{K,V}
    k::Int
    key::Function
    keys::Vector{K}
    values::Vector{V}
end

function KMin(k::Integer, key::Function)::KMin{Any,Any}
    return KMin{Any,Any}(Int(k), key, Any[], Any[])
end

function Base.push!(km::KMin, value)::Nothing
    key = km.key(value)
    if length(km.values) == km.k && key > km.keys[end]
        return nothing
    end
    i = searchsortedlast(km.keys, key) + 1
    insert!(km.keys, i, key)
    insert!(km.values, i, value)
    if length(km.values) > km.k
        pop!(km.keys)
        pop!(km.values)
    end
    return nothing
end

Base.length(km::KMin)::Int = length(km.values)
Base.iterate(km::KMin) = iterate(km.values)
Base.iterate(km::KMin, state) = iterate(km.values, state)

"""
    beam_search(problem::Problem; solution::Union{Nothing, Solution}=nothing, bw::Integer=10) -> solution::Solution

Beam search: a constructive heuristic that maintains a beam of width `bw`
partial solutions, extending each by the best moves at each step.
"""
function beam_search(problem::Problem; solution::Union{Nothing, Solution}=nothing, bw::Integer=10)::Solution
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

    v::Vector{Tuple{<:Real, Solution}} = [(lb, best)]

    while true
        candidates = KMin(Int(bw), x -> x[1])
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

        v = Tuple{<:Real, Solution}[]
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
