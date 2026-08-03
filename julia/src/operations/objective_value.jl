"""
    objective_value(solution::Solution) -> value::Union{Nothing, <:Real}

Return the objective value of `solution`, or `nothing` if the solution
is not feasible.
"""
function objective_value(::Solution)::Union{Nothing, <:Real} end
