"""
    Problem

Abstract type representing an optimisation problem.
A problem should implement `empty_solution`, and optionally
`construction_neighbourhood`, `local_neighbourhood`, `destruction_neighbourhood`,
`heuristic_solution`, and `random_solution`.
"""
abstract type Problem end
