"""
    random_move(neighbourhood::Neighbourhood, solution::Solution) -> move::Union{Nothing, Move}

Return a single random `Move` from `neighbourhood` for `solution`,
or `nothing` if no move exists.
"""
function random_move(::Neighbourhood, ::Solution)::Union{Nothing, Move} end
