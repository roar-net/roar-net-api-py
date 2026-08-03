"""
    lower_bound(solution::Solution) -> value::Union{Nothing, <:Real}

Return a lower bound on the objective value of `solution`.
Used by beam search and other constructive algorithms.
"""
function lower_bound(::Solution)::Union{Nothing, <:Real} end
