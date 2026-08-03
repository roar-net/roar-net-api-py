"""
    random_moves_without_replacement(neighbourhood::Neighbourhood, solution::Solution) -> iterable::Any

Return an iterable of `Move`s from `neighbourhood` for `solution` in random
order without replacement. The return type is declared `Any` because
concrete implementations may use either a `Vector{<:Move}` or a
`Channel{<:Move}`.
"""
function random_moves_without_replacement(::Neighbourhood, ::Solution)::Any end
