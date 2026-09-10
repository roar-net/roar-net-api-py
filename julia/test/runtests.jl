
# SPDX-FileCopyrightText: © 2025 Authors of the roar-net-api-py project <https://github.com/roar-net/roar-net-api-py/blob/main/AUTHORS>
#
# SPDX-License-Identifier: Apache-2.0

using Test
using RoarNetAPI

@testset "RoarNetAPI" begin
    @testset "Interface functions throw MethodError" begin
        @test_throws MethodError apply_move(nothing, nothing)
        @test_throws MethodError empty_solution(nothing)
        @test_throws MethodError objective_value(nothing)
        @test_throws MethodError copy_solution(nothing)
        @test_throws MethodError lower_bound(nothing)
        @test_throws MethodError moves(nothing, nothing)
        @test_throws MethodError local_neighbourhood(nothing)
        @test_throws MethodError construction_neighbourhood(nothing)
        @test_throws MethodError random_solution(nothing)
        @test_throws MethodError lower_bound_increment(nothing, nothing)
        @test_throws MethodError objective_value_increment(nothing, nothing)
        @test_throws MethodError random_move(nothing, nothing)
        @test_throws MethodError random_moves_without_replacement(nothing, nothing)
    end

@testset "Abstract types" begin
        @test Problem <: Any
        @test Solution <: Any
        @test Move <: Any
        @test Neighbourhood <: Any
    end

    @testset "Strong typing of operations" begin
        # Each interface operation should be registered with a method whose
        # arguments are constrained to the abstract types exported by
        # RoarNetAPI.Types. This guards the contract that concrete types
        # subtype these markers.
        @test hasmethod(apply_move, Tuple{Move, Solution})
        @test hasmethod(construction_neighbourhood, Tuple{Problem})
        @test hasmethod(copy_solution, Tuple{Solution})
        @test hasmethod(destruction_neighbourhood, Tuple{Problem})
        @test hasmethod(empty_solution, Tuple{Problem})
        @test hasmethod(heuristic_solution, Tuple{Problem})
        @test hasmethod(invert_move, Tuple{Move})
        @test hasmethod(local_neighbourhood, Tuple{Problem})
        @test hasmethod(lower_bound, Tuple{Solution})
        @test hasmethod(lower_bound_increment, Tuple{Move, Solution})
        @test hasmethod(moves, Tuple{Neighbourhood, Solution})
        @test hasmethod(objective_value, Tuple{Solution})
        @test hasmethod(objective_value_increment, Tuple{Move, Solution})
        @test hasmethod(random_move, Tuple{Neighbourhood, Solution})
        @test hasmethod(random_moves_without_replacement, Tuple{Neighbourhood, Solution})
        @test hasmethod(random_solution, Tuple{Problem})

        # Algorithm entry points
        @test hasmethod(greedy_construction, Tuple{Problem})
        @test hasmethod(greedy_construction_with_random_tie_breaking, Tuple{Problem})
        @test hasmethod(beam_search, Tuple{Problem})
        @test hasmethod(best_improvement, Tuple{Problem, Solution})
        @test hasmethod(first_improvement, Tuple{Problem, Solution})
        @test hasmethod(grasp, Tuple{Problem, Real})
        @test hasmethod(rls, Tuple{Problem, Solution, Real})
        @test hasmethod(sa, Tuple{Problem, Solution, Real, Real})
    end

    @testset "Greedy construction on minimal problem" begin
        struct MinimalProblem <: Problem end
        struct MinimalSolution <: Solution
            value::Int
        end
        struct EmptyNeighbourhood <: Neighbourhood end

        function RoarNetAPI.empty_solution(::MinimalProblem)::MinimalSolution
            return MinimalSolution(0)
        end
        function RoarNetAPI.objective_value(sol::MinimalSolution)::Int
            return sol.value
        end
        function RoarNetAPI.copy_solution(sol::MinimalSolution)::MinimalSolution
            return MinimalSolution(sol.value)
        end
        function RoarNetAPI.construction_neighbourhood(::MinimalProblem)::EmptyNeighbourhood
            return EmptyNeighbourhood()
        end
        function RoarNetAPI.moves(::EmptyNeighbourhood, ::MinimalSolution)::Vector{Any}
            return []
        end

        prob = MinimalProblem()
        sol = greedy_construction(prob)
        @test objective_value(sol) == 0
    end

    @testset "Beam search KMin" begin
        km = RoarNetAPI.Algorithms.KMin(3, x -> x)
        for v in [5, 3, 7, 1, 9, 2, 8, 4, 6]
            push!(km, v)
        end
        @test length(km) == 3
        @test collect(km) == [1, 2, 3]
    end

    @testset "Simulated annealing helpers" begin
        schedule = RoarNetAPI.Algorithms.LinearDecay(100.0)
        @test schedule(1.0) == 100.0
        @test schedule(0.5) == 50.0

        acc = RoarNetAPI.Algorithms.ExponentialAcceptance()
        @test acc(-1.0, 10.0) == 1.0
        @test acc(0.0, 10.0) == 1.0
        @test 0 < acc(1.0, 10.0) < 1
    end

    @testset "TSP example integration" begin
        include("../examples/tsp/tsp.jl")

        buf = IOBuffer("""
NAME : test5
TYPE : TSP
DIMENSION : 5
EDGE_WEIGHT_TYPE : EUC_2D
NODE_COORD_SECTION
1 0 0
2 1 0
3 1 1
4 0 1
5 2 2
EOF
""")
        prob = from_textio(buf)
        @test prob.n == 5
        @test prob.name == "test5"
        @test prob.dist[1, 2] == 1
        @test prob.dist[1, 3] == round(Int, sqrt(2))

        sol = greedy_construction(prob)
        obj = objective_value(sol)
        @test obj !== nothing
        @test obj isa Int
        @test is_feasible(sol)

        improved = best_improvement(prob, sol)
        @test objective_value(improved) <= objective_value(sol)

        improved2 = first_improvement(prob, sol)
        @test objective_value(improved2) <= objective_value(sol)
    end
end
