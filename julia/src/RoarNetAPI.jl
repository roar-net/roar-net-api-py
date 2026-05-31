module RoarNetAPI

module Operations
    include("operations/apply_move.jl")
    include("operations/construction_neighbourhood.jl")
    include("operations/copy_solution.jl")
    include("operations/destruction_neighbourhood.jl")
    include("operations/empty_solution.jl")
    include("operations/heuristic_solution.jl")
    include("operations/invert_move.jl")
    include("operations/local_neighbourhood.jl")
    include("operations/lower_bound.jl")
    include("operations/lower_bound_increment.jl")
    include("operations/moves.jl")
    include("operations/objective_value.jl")
    include("operations/objective_value_increment.jl")
    include("operations/random_move.jl")
    include("operations/random_moves_without_replacement.jl")
    include("operations/random_solution.jl")

    export apply_move,
        construction_neighbourhood,
        copy_solution,
        destruction_neighbourhood,
        empty_solution,
        heuristic_solution,
        invert_move,
        local_neighbourhood,
        lower_bound,
        lower_bound_increment,
        moves,
        objective_value,
        objective_value_increment,
        random_move,
        random_moves_without_replacement,
        random_solution
end

module Types
    using ..Operations

    include("types/Move.jl")
    include("types/Neighbourhood.jl")
    include("types/Problem.jl")
    include("types/Solution.jl")

    export Move,
        Neighbourhood,
        Problem,
        Solution
end

module Algorithms
    using ..Operations

    include("algorithms/greedy_construction.jl")
    include("algorithms/beam_search.jl")
    include("algorithms/best_improvement.jl")
    include("algorithms/first_improvement.jl")
    include("algorithms/grasp.jl")
    include("algorithms/rls.jl")
    include("algorithms/sa.jl")

    export greedy_construction,
        greedy_construction_with_random_tie_breaking,
        beam_search,
        best_improvement,
        first_improvement,
        grasp,
        rls,
        sa
end

# Re-export all public names from submodules
using .Operations: apply_move, construction_neighbourhood, copy_solution,
    destruction_neighbourhood, empty_solution, heuristic_solution, invert_move,
    local_neighbourhood, lower_bound, lower_bound_increment, moves,
    objective_value, objective_value_increment, random_move,
    random_moves_without_replacement, random_solution

using .Types: Move, Neighbourhood, Problem, Solution

using .Algorithms: greedy_construction, greedy_construction_with_random_tie_breaking,
    beam_search, best_improvement, first_improvement, grasp, rls, sa

export apply_move, construction_neighbourhood, copy_solution,
    destruction_neighbourhood, empty_solution, heuristic_solution, invert_move,
    local_neighbourhood, lower_bound, lower_bound_increment, moves,
    objective_value, objective_value_increment, random_move,
    random_moves_without_replacement, random_solution,
    Move, Neighbourhood, Problem, Solution,
    greedy_construction, greedy_construction_with_random_tie_breaking,
    beam_search, best_improvement, first_improvement, grasp, rls, sa

end
