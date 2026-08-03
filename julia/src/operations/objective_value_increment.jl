"""
    objective_value_increment(move::Move, solution::Solution) -> value::Union{Nothing, <:Real}

Return the objective value increment for applying `move` to `solution`,
or `nothing` if the move is not applicable. Used by local search algorithms.
"""
function objective_value_increment(::Move, ::Solution)::Union{Nothing, <:Real} end
