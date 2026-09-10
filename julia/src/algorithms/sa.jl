# SPDX-FileCopyrightText: © 2025 Authors of the roar-net-api-py project <https://github.com/roar-net/roar-net-api-py/blob/main/AUTHORS>
#
# SPDX-License-Identifier: Apache-2.0
"""
    LinearDecay{T <: Real}

A temperature schedule that linearly decays from an initial temperature.
"""
struct LinearDecay{T<:Real}
    init_temp::T
end

function (s::LinearDecay)(t::Real)::Float64
    return Float64(t * s.init_temp)
end

"""
    ExponentialAcceptance

An acceptance probability function for simulated annealing.
Returns 1.0 for non-worsening moves, and `exp(-incr / t)` for worsening moves.
"""
struct ExponentialAcceptance end

function (::ExponentialAcceptance)(incr::Real, t::Real)::Float64
    if incr <= 0
        return 1.0
    else
        return exp(-incr / t)
    end
end

"""
    sa(problem::Problem, solution::Solution, budget::Real, init_temp::Real;
       temperature::Union{Nothing, Function}=nothing,
       acceptance::Union{Nothing, Function}=nothing) -> solution::Solution

Simulated Annealing: improve `solution` using `budget` seconds of computation,
with initial temperature `init_temp`. Optional `temperature` schedule and
`acceptance` probability function can be provided.
"""
function sa(problem::Problem, solution::Solution, budget::Real, init_temp::Real;
        temperature::Union{Nothing, Function}=nothing,
        acceptance::Union{Nothing, Function}=nothing)::Solution

    if temperature === nothing
        temperature = LinearDecay(Float64(init_temp))
    end
    if acceptance === nothing
        acceptance = ExponentialAcceptance()
    end

    start = time()
    neigh = local_neighbourhood(problem)
    best = copy_solution(solution)
    best_obj = objective_value(best)

    while time() - start < budget
        for mv in random_moves_without_replacement(neigh, solution)
            t = temperature(1 - (time() - start) / budget)
            if t <= 0
                break
            end
            incr = objective_value_increment(mv, solution)
            @assert incr !== nothing

            if acceptance(incr, t) >= rand()
                solution = apply_move(mv, solution)
                obj = objective_value(solution)
                @assert obj !== nothing

                if best_obj === nothing || obj < best_obj
                    @info "Best solution: $obj"
                    best = copy_solution(solution)
                    best_obj = obj
                end
                break
            end
        end
    end

    return best
end
