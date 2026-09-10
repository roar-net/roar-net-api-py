<!--
SPDX-FileCopyrightText: © 2025 Authors of the roar-net-api-py project <https://github.com/roar-net/roar-net-api-py/blob/main/AUTHORS>

SPDX-License-Identifier: Apache-2.0
-->

# ROAR-NET API Julia Tutorial

This tutorial walks you through using the ROAR-NET API in Julia, from
installation to implementing your own optimisation model.

## 0. Strong typing

The library is strongly typed throughout. Every interface operation
declares a typed signature against the abstract markers (`Problem`,
`Solution`, `Move`, `Neighbourhood`); every algorithm entry point
takes its problem/solution arguments against those same markers. The
return contracts are:

- Solution-returning operations: `::Solution`.
- Numeric returns that may be infeasible: `::Union{Nothing, <:Real}`.
- Iterable returns: `::Any` (either a `Vector{<:Move}` or a
  `Channel{<:Move}` is accepted).

When you implement your own model, mirror these contracts on your
concrete methods — add the return type that matches your concrete
types (e.g. `::Int` for knapsack, `::Float64` for continuous problems)
and subtype the appropriate abstract marker.

## 1. Installation

The package is in the `julia/` directory of this repository. To use it:

```bash
git clone <repo-url>
cd roar-net-api-py
```

Then from Julia:

```julia
julia> using Pkg; Pkg.develop(path="julia")
```

Or run scripts with `--project`:

```bash
julia --project=julia my_script.jl
```

## 2. Quick start: TSP example

The repository includes a complete TSP (Travelling Salesman Problem)
example. To run it on a sample instance:

```bash
# Create a test instance
cat > /tmp/test.tsp << 'EOF'
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

#For the first time run add all the depencencies

julia --project=julia -e 'using Pkg; Pkg.instantiate()'
# Run the solver
julia --project=julia julia/examples/tsp/tsp.jl < /tmp/test.tsp
```

Expected output (stderr):
```
[ Info: Objective value after constructive search: 9
[ Info: Objective value after local search: 8
```

And to stdout, a TSPLIB-format tour file.

You can also download real TSPLIB instances from
[tsplib](http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/) and
run them:

```bash
julia --project=julia julia/examples/tsp/tsp.jl < instances/berlin52.tsp
```

## 3. Implementing your own model

To use the algorithms with your own optimisation problem, you need to
implement the interface functions for your concrete types. Here is a
step-by-step guide with a simple knapsack problem example.

### 3.1. Define your types

Create a file `my_model.jl`:

```julia
using RoarNetAPI
using Random

# Your concrete types, subtyping the abstract types
struct KnapsackItem
    weight::Int
    value::Int
end

mutable struct MyKnapsack <: Problem
    capacity::Int
    items::Vector{KnapsackItem}
end

mutable struct MySolution <: Solution
    selected::Vector{Bool}
    weight::Int
    value::Int
end
```

### 3.2. Implement the interface functions

You need to implement at minimum the functions required by the
algorithms you want to use. For greedy construction (constructive
neighbourhood), implement:

```julia
# Define a constructive neighbourhood type
struct MyKnapsackNeighbourhood <: Neighbourhood
    problem::MyKnapsack
end

# Create an empty solution
function RoarNetAPI.empty_solution(prob::MyKnapsack)::MySolution
    return MySolution(
        falses(length(prob.items)),
        0, 0
    )
end

# Return a constructive neighbourhood
function RoarNetAPI.construction_neighbourhood(prob::MyKnapsack)::MyKnapsackNeighbourhood
    return MyKnapsackNeighbourhood(prob)
end

# Define a move type
struct MyAddMove <: Move
    neighbourhood::MyKnapsackNeighbourhood
    item_idx::Int
end

# List all valid moves from a solution
function RoarNetAPI.moves(neigh::MyKnapsackNeighbourhood, sol::MySolution)::Vector{MyAddMove}
    prob = neigh.problem
    moves = MyAddMove[]
    for i in eachindex(prob.items)
        if !sol.selected[i] && sol.weight + prob.items[i].weight <= prob.capacity
            push!(moves, MyAddMove(neigh, i))
        end
    end
    return moves
end

# Apply a move to a solution
function RoarNetAPI.apply_move(mv::MyAddMove, sol::MySolution)::MySolution
    prob = mv.neighbourhood.problem
    item = prob.items[mv.item_idx]
    sol.selected[mv.item_idx] = true
    sol.weight += item.weight
    sol.value += item.value
    return sol
end

# Lower bound increment (for constructive guidance)
function RoarNetAPI.lower_bound_increment(mv::MyAddMove, sol::MySolution)::Int
    prob = mv.neighbourhood.problem
    item = prob.items[mv.item_idx]
    # For knapsack, lower bound on remaining value is 0
    # The "increment" is the negative value (we minimise negative value)
    return -item.value
end

# Objective value
function RoarNetAPI.objective_value(sol::MySolution)::Int
    return -sol.value
end

# Copy a solution
function RoarNetAPI.copy_solution(sol::MySolution)::MySolution
    return MySolution(copy(sol.selected), sol.weight, sol.value)
end
```

### 3.3. Run algorithms

```julia
# Create a problem instance
items = [KnapsackItem(2, 3), KnapsackItem(3, 4), KnapsackItem(4, 5),
         KnapsackItem(5, 7), KnapsackItem(9, 10)]
prob = MyKnapsack(15, items)

# Solve with greedy construction
solution = greedy_construction(prob)
println("Selected: ", findall(solution.selected))
println("Value: ", -objective_value(solution))

# OR use beam search
solution = beam_search(prob; bw=5)
println("Beam search selected: ", findall(solution.selected))
println("Beam search value: ", -objective_value(solution))
```

## 4. Using local search

For local search algorithms (best improvement, first improvement, RLS,
SA), you also need a local neighbourhood:

```julia
struct MyLocalNeighbourhood <: Neighbourhood
    problem::MyKnapsack
end

function RoarNetAPI.local_neighbourhood(prob::MyKnapsack)::MyLocalNeighbourhood
    return MyLocalNeighbourhood(prob)
end

struct MySwapMove <: Move
    neighbourhood::MyLocalNeighbourhood
    add_idx::Int
    remove_idx::Int
end

function RoarNetAPI.moves(neigh::MyLocalNeighbourhood, sol::MySolution)::Vector{MySwapMove}
    prob = neigh.problem
    moves = MySwapMove[]
    for add in eachindex(prob.items)
        if !sol.selected[add]
            for remove in eachindex(prob.items)
                if sol.selected[remove]
                    new_weight = sol.weight - prob.items[remove].weight + prob.items[add].weight
                    if new_weight <= prob.capacity
                        push!(moves, MySwapMove(neigh, add, remove))
                    end
                end
            end
        end
    end
    return moves
end

function RoarNetAPI.apply_move(mv::MySwapMove, sol::MySolution)::MySolution
    prob = mv.neighbourhood.problem
    sol.selected[mv.remove_idx] = false
    sol.selected[mv.add_idx] = true
    sol.weight += prob.items[mv.add_idx].weight - prob.items[mv.remove_idx].weight
    sol.value += prob.items[mv.add_idx].value - prob.items[mv.remove_idx].value
    return sol
end

function RoarNetAPI.objective_value_increment(mv::MySwapMove, sol::MySolution)::Int
    prob = mv.neighbourhood.problem
    return -(prob.items[mv.add_idx].value - prob.items[mv.remove_idx].value)
end
```

Then use local search:

```julia
solution = greedy_construction(prob)
solution = best_improvement(prob, solution)
# or first_improvement(prob, solution)
# or rls(prob, solution, 5.0)  # 5 seconds budget
# or sa(prob, solution, 5.0, 10.0)  # 5s, initial temp 10
```

## 5. Available algorithms

| Algorithm | Function | Type | Budget? | Key operations needed |
|---|---|---|---|---|
| Greedy construction | `greedy_construction(problem)` | Constructive | No | `construction_neighbourhood`, `moves`, `lower_bound_increment`, `apply_move` |
| Beam search | `beam_search(problem; bw=10)` | Constructive | No | Same + `lower_bound`, `copy_solution` |
| GRASP | `grasp(problem, budget; alpha=0.1)` | Constructive | Yes (time) | Same as greedy + `copy_solution` |
| Best improvement | `best_improvement(problem, solution)` | Local search | No | `local_neighbourhood`, `moves`, `objective_value_increment`, `apply_move` |
| First improvement | `first_improvement(problem, solution)` | Local search | No | Replace `moves` with `random_moves_without_replacement` |
| RLS | `rls(problem, solution, budget)` | Local search | Yes (time) | Same as first improvement |
| Simulated annealing | `sa(problem, solution, budget, init_temp)` | Local search | Yes (time) | + `copy_solution` |

## 6. Algorithm reference

### Greedy construction

Builds a solution by repeatedly applying the move with the smallest
lower bound increment. Stops when no moves remain.

```julia
greedy_construction(problem::Problem; solution::Union{Nothing, Solution}=nothing)::Solution
```

### Beam search

Maintains a beam of `bw` partial solutions, extending each by its best
moves at each step. Returns the best feasible solution found.

```julia
beam_search(problem::Problem; solution::Union{Nothing, Solution}=nothing, bw::Integer=10)::Solution
```

### GRASP

Greedy Randomized Adaptive Search Procedure. Builds solutions
iteratively within a time budget using a Restricted Candidate List
controlled by `alpha`. Optionally applies a local search to each
constructed solution.

```julia
grasp(problem::Problem, budget::Real;
      solution::Union{Nothing, Solution}=nothing,
      alpha::Real=0.1,
      local_search::Union{Nothing, Function}=nothing)::Solution
```

### Best improvement

Exhaustively evaluates all moves in the local neighbourhood, applies the
best improving move, and repeats until no improving move exists.

```julia
best_improvement(problem::Problem, solution::Solution)::Solution
```

### First improvement

Scans the local neighbourhood in random order, applies the first
improving move found, and repeats until no improving move exists.

```julia
first_improvement(problem::Problem, solution::Solution)::Solution
```

### RLS (Random Local Search)

Samples random moves from the local neighbourhood within a time budget.
Applies any non-worsening move immediately.

```julia
rls(problem::Problem, solution::Solution, budget::Real)::Solution
```

### SA (Simulated Annealing)

Samples random moves with acceptance probability
$\exp(-\Delta / T)$ for worsening moves. Temperature decays linearly
from `init_temp` to 0 over the `budget`.

```julia
sa(problem::Problem, solution::Solution, budget::Real, init_temp::Real;
   temperature::Union{Nothing, Function}=nothing,
   acceptance::Union{Nothing, Function}=nothing)::Solution
```

Custom temperature schedules and acceptance functions can be provided as
callable objects:

```julia
sa(problem, sol, 10.0, 100.0,
    temperature=t -> t * 100.0,  # linear decay
    acceptance=(incr, t) -> incr <= 0 ? 1.0 : exp(-incr / t))
```

## 7. Tips

- **Mutable structs**: Solutions are typically `mutable struct` so that
  `apply_move` can modify them in-place.
- **Strong typing**: Always annotate the fields of your structs and add
  explicit return types to your interface methods (e.g.
  `objective_value(sol::MySolution)::Int`). This keeps dispatch
  type-stable and lets the `hasmethod` tests in the test suite verify
  that every operation is implemented.
- **Performance**: Use `Vector` for dense data and `Dict` for sparse.
  For the TSP distance matrix, a `Matrix{Int}` is fastest.
- **Randomness**: Call `Random.seed!(n)` for reproducible runs.
- **Logging**: The algorithms use `@info` for progress messages.
  Control verbosity with:
  ```julia
  using Logging
  disable_logging(Logging.Info)  # suppress info messages
  ```
