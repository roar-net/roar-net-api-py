#!/usr/bin/env julia

using RoarNetAPI
using Random

include("tsp.jl")

function solve(algo, prob, budget)
    if algo == "greedy"
        return greedy_construction(prob)
    elseif algo == "best"
        s = greedy_construction(prob)
        return best_improvement(prob, s)
    elseif algo == "first"
        s = greedy_construction(prob)
        return first_improvement(prob, s)
    elseif algo == "beam"
        return beam_search(prob, bw=10)
    elseif algo == "grasp"
        return grasp(prob, budget)
    elseif algo == "rls"
        s = greedy_construction(prob)
        return rls(prob, s, budget)
    elseif algo == "sa"
        s = greedy_construction(prob)
        return sa(prob, s, budget, 30.0)
    else
        error("Unknown algorithm: $algo")
    end
end

function run_benchmark()
    if length(ARGS) < 3
        println(stderr, "Usage: benchmark.jl <algorithm> <seed> <budget> [trials]")
        println(stderr, "  algorithm: greedy, best, first, beam, grasp, rls, sa")
        exit(1)
    end

    algo = ARGS[1]
    seed0 = parse(Int, ARGS[2])
    budget = parse(Float64, ARGS[3])
    trials = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 1

    # Read instance once (stdin is consumed line-by-line).
    # Store in a buffer so we can re-read for each trial.
    instance_lines = readlines(stdin)
    instance_text = join(instance_lines, "\n")

    for t in 0:trials-1
        s = seed0 + t
        Random.seed!(s)

        # Re-parse the instance for each trial (stdin consumed)
        buf = IOBuffer(instance_text)
        prob = from_textio(buf)

        start = time()
        sol = solve(algo, prob, budget)
        elapsed = time() - start

        obj = objective_value(sol)
        obj_str = obj === nothing ? "none" : string(obj)

        println("$s OBJ=$obj_str TIME=$elapsed")
    end
end

run_benchmark()
