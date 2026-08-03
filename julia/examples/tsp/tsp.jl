#!/usr/bin/env julia

using RoarNetAPI
using Random

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

argmin(seq::AbstractVector)::Int = findmin(seq)[2]

function sparse_fisher_yates_iter(n::Int)::Channel{Int}
    return Channel{Int}() do ch
        p = Dict{Int,Int}()
        for i in n-1:-1:0
            r = rand(0:i)
            put!(ch, get(p, r, r))
            if i != r
                p[r] = get(p, i, i)
            end
        end
    end
end

# ---------------------------------------------------------------------------
# Problem (defined first to avoid circular deps)
# ---------------------------------------------------------------------------

mutable struct TSPProblem <: Problem
    dist::Matrix{Int}
    name::String
    n::Int
end

function TSPProblem(dist::Matrix{Int}, name::String)::TSPProblem
    n::Int = size(dist, 1)
    return TSPProblem(dist, name, n)
end

function Base.show(io::IO, prob::TSPProblem)::Nothing
    for i in 1:prob.n
        println(io, join(prob.dist[i, :], " "))
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Neighbourhoods (defined before moves that reference them)
# ---------------------------------------------------------------------------

struct AddNeighbourhood <: Neighbourhood
    problem::TSPProblem
end

struct TwoOptNeighbourhood <: Neighbourhood
    problem::TSPProblem
end

# ---------------------------------------------------------------------------
# Solution
# ---------------------------------------------------------------------------

mutable struct TSPSolution <: Solution
    problem::TSPProblem
    tour::Vector{Int}
    not_visited::Set{Int}
    lb::Int
end

Base.show(io::IO, sol::TSPSolution)::Nothing = print(io, join(sol.tour, " "))
is_feasible(sol::TSPSolution)::Bool = isempty(sol.not_visited)

function to_textio(sol::TSPSolution, io::IO)::Nothing
    println(io, "NAME : $(sol.problem.name).tour")
    println(io, "TYPE : TOUR")
    println(io, "DIMENSION : $(sol.problem.n)")
    println(io, "TOUR_SECTION")
    for city in sol.tour
        println(io, city)
    end
    println(io, "EOF")
    return nothing
end

function RoarNetAPI.copy_solution(sol::TSPSolution)::TSPSolution
    return TSPSolution(sol.problem, copy(sol.tour), copy(sol.not_visited), sol.lb)
end

function RoarNetAPI.objective_value(sol::TSPSolution)::Union{Nothing, Int}
    if is_feasible(sol)
        return sol.lb
    end
    return nothing
end

function RoarNetAPI.lower_bound(sol::TSPSolution)::Int
    return sol.lb
end

# ---------------------------------------------------------------------------
# Moves
# ---------------------------------------------------------------------------

struct AddMove <: Move
    neighbourhood::AddNeighbourhood
    i::Int
    j::Int
end

struct TwoOptMove <: Move
    neighbourhood::TwoOptNeighbourhood
    ix::Int
    jx::Int
end

# -- AddMove ---------------------------------------------------------------

function RoarNetAPI.apply_move(mv::AddMove, sol::TSPSolution)::TSPSolution
    @assert sol.tour[end] == mv.i
    prob = sol.problem
    sol.lb += prob.dist[mv.i, mv.j]
    if length(sol.not_visited) == 1
        sol.lb += prob.dist[mv.j, sol.tour[1]]
    end
    push!(sol.tour, mv.j)
    delete!(sol.not_visited, mv.j)
    return sol
end

function RoarNetAPI.lower_bound_increment(mv::AddMove, sol::TSPSolution)::Int
    @assert sol.tour[end] == mv.i
    prob = sol.problem
    incr::Int = prob.dist[mv.i, mv.j]
    if length(sol.not_visited) == 1
        incr += prob.dist[mv.j, sol.tour[1]]
    end
    return incr
end

# -- TwoOptMove ------------------------------------------------------------

function RoarNetAPI.apply_move(mv::TwoOptMove, sol::TSPSolution)::TSPSolution
    prob = sol.problem
    n, ix, jx = prob.n, mv.ix, mv.jx
    t = sol.tour
    sol.lb -= prob.dist[t[ix], t[ix+1]] + prob.dist[t[jx], t[(jx % n) + 1]]
    sol.lb += prob.dist[t[ix], t[jx]] + prob.dist[t[ix+1], t[(jx % n) + 1]]
    sol.tour[ix+1:jx] = reverse(sol.tour[ix+1:jx])
    return sol
end

function RoarNetAPI.objective_value_increment(mv::TwoOptMove, sol::TSPSolution)::Int
    prob = sol.problem
    n, ix, jx = prob.n, mv.ix, mv.jx
    t = sol.tour
    incr::Int = prob.dist[t[ix], t[jx]] + prob.dist[t[ix+1], t[(jx % n) + 1]]
    incr -= prob.dist[t[ix], t[ix+1]] + prob.dist[t[jx], t[(jx % n) + 1]]
    return incr
end

# ---------------------------------------------------------------------------
# Neighbourhood implementations
# ---------------------------------------------------------------------------

function RoarNetAPI.moves(neigh::AddNeighbourhood, sol::TSPSolution)::Vector{AddMove}
    @assert neigh.problem == sol.problem
    i::Int = sol.tour[end]
    return AddMove[AddMove(neigh, i, j) for j in sort(collect(sol.not_visited))]
end

function RoarNetAPI.moves(neigh::TwoOptNeighbourhood, sol::TSPSolution)::Vector{TwoOptMove}
    @assert neigh.problem == sol.problem
    n::Int = neigh.problem.n
    @assert is_feasible(sol)
    result::Vector{TwoOptMove} = TwoOptMove[]
    for ix in 1:n-2
        end_val::Int = ix == 1 ? n-1 : n
        for jx in ix+2:end_val
            push!(result, TwoOptMove(neigh, ix, jx))
        end
    end
    return result
end

function RoarNetAPI.random_moves_without_replacement(neigh::TwoOptNeighbourhood, sol::TSPSolution)::Channel{TwoOptMove}
    @assert neigh.problem == sol.problem
    n::Int = neigh.problem.n
    @assert is_feasible(sol)
    return Channel{TwoOptMove}(32) do ch
        for x in sparse_fisher_yates_iter(n * (n - 3) ÷ 2)
            jx::Int = (1 + isqrt(1 + 8 * x)) ÷ 2
            ix::Int = x - jx * (jx - 1) ÷ 2 + 1
            jx += 2
            if ix == 1 && jx == n
                ix = n - 2
            end
            put!(ch, TwoOptMove(neigh, ix, jx))
        end
    end
end

function RoarNetAPI.random_move(neigh::TwoOptNeighbourhood, sol::TSPSolution)::Union{Nothing, TwoOptMove}
    for mv in random_moves_without_replacement(neigh, sol)
        return mv
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Problem implementations
# ---------------------------------------------------------------------------

function RoarNetAPI.construction_neighbourhood(prob::TSPProblem)::AddNeighbourhood
    return AddNeighbourhood(prob)
end

function RoarNetAPI.local_neighbourhood(prob::TSPProblem)::TwoOptNeighbourhood
    return TwoOptNeighbourhood(prob)
end

function RoarNetAPI.empty_solution(prob::TSPProblem)::TSPSolution
    return TSPSolution(prob, [1], Set(2:prob.n), 0)
end

function RoarNetAPI.random_solution(prob::TSPProblem)::TSPSolution
    c::Vector{Int} = collect(2:prob.n)
    shuffle!(c)
    pushfirst!(c, 1)
    obj::Int = prob.dist[c[end], c[1]]
    for ix in 2:prob.n
        obj += prob.dist[c[ix-1], c[ix]]
    end
    return TSPSolution(prob, c, Set{Int}(), obj)
end

function from_textio(io::IO)::TSPProblem
    n::Union{Nothing, Int} = nothing
    dt::Union{Nothing, String} = nothing
    name::String = "unnamed"::String

    line = strip(readline(io))
    while line != "NODE_COORD_SECTION" && !eof(io)
        parts = split(line, ":", limit=2)
        key = strip(parts[1])
        if key == "DIMENSION"
            n = parse(Int, strip(parts[2]))
        elseif key == "EDGE_WEIGHT_TYPE"
            dt = strip(parts[2])
        elseif key == "NAME"
            name = String(strip(parts[2]))
        end
        line = strip(readline(io))
    end

    if n !== nothing && dt == "EUC_2D"
        kxy = [Tuple(parse.(Float64, split(readline(io)))) for _ in 1:n]
        sort!(kxy, by = x -> x[1])
        dist = zeros(Int, n, n)
        for i in 1:n
            if kxy[i][1] != i
                error("Invalid instance")
            end
            for j in 1:n
                dist[i, j] = floor(Int, sqrt((kxy[i][2] - kxy[j][2])^2 + (kxy[i][3] - kxy[j][3])^2) + 0.5)
            end
        end
        return TSPProblem(dist, name)
    else
        error("Instance format $dt not supported")
    end
end

# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

function main()::Nothing
    problem = from_textio(stdin)

    solution = greedy_construction(problem)
    @info "Objective value after constructive search: $(objective_value(solution))"

    solution = sa(problem, solution, 10.0, 30.0)
    @info "Objective value after local search: $(objective_value(solution))"

    to_textio(solution, stdout)
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
