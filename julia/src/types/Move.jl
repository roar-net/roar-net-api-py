"""
    Move

Abstract type representing a move that can be applied to a solution.
A move should implement `apply_move`, and optionally
`lower_bound_increment`, `objective_value_increment`, and `invert_move`.
"""
abstract type Move end
