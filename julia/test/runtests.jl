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

    @testset "Greedy construction on minimal problem" begin
        struct MinimalProblem <: Problem end
        struct MinimalSolution <: Solution
            value::Int
        end
        struct EmptyNeighbourhood <: Neighbourhood end

        function RoarNetAPI.empty_solution(::MinimalProblem)
            return MinimalSolution(0)
        end
        function RoarNetAPI.objective_value(sol::MinimalSolution)
            return sol.value
        end
        function RoarNetAPI.copy_solution(sol::MinimalSolution)
            return MinimalSolution(sol.value)
        end
        function RoarNetAPI.construction_neighbourhood(::MinimalProblem)
            return EmptyNeighbourhood()
        end
        function RoarNetAPI.moves(::EmptyNeighbourhood, ::MinimalSolution)
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
