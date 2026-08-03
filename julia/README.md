<!--
SPDX-FileCopyrightText: © 2025 Authors of the roar-net-api-py project <https://github.com/roar-net/roar-net-api-py/blob/main/AUTHORS>

SPDX-License-Identifier: Apache-2.0
-->

# ROAR-NET API - Julia

This repository contains a Julia implementation of the ROAR-NET API
Specification and several algorithms. It is a port of the
[roar-net-api-py](https://github.com/roar-net/roar-net-api-py) Python
library. You can find the full specification in the
[roar-net-api-spec](https://github.com/roar-net/roar-net-api-spec)
repository.

## Contents

This library provides protocol types (as Julia abstract types and
interface functions) for all operations in the specification, and
several metaheuristic algorithms.

### Interface operations

Interface functions are defined in the `RoarNetAPI.Operations` module
(and re-exported). Each function declares a typed contract that concrete
types must implement via multiple dispatch:

| Operation | Julia signature | Return type |
|---|---|---|
| `SupportsApplyMove` | `apply_move(::Move, ::Solution)` | `Solution` |
| `SupportsConstructionNeighbourhood` | `construction_neighbourhood(::Problem)` | `Neighbourhood` |
| `SupportsCopySolution` | `copy_solution(::Solution)` | `Solution` |
| `SupportsDestructionNeighbourhood` | `destruction_neighbourhood(::Problem)` | `Neighbourhood` |
| `SupportsEmptySolution` | `empty_solution(::Problem)` | `Solution` |
| `SupportsHeuristicSolution` | `heuristic_solution(::Problem)` | `Solution` |
| `SupportsInvertMove` | `invert_move(::Move)` | `Move` |
| `SupportsLocalNeighbourhood` | `local_neighbourhood(::Problem)` | `Neighbourhood` |
| `SupportsLowerBound` | `lower_bound(::Solution)` | `Union{Nothing, <:Real}` |
| `SupportsLowerBoundIncrement` | `lower_bound_increment(::Move, ::Solution)` | `Union{Nothing, <:Real}` |
| `SupportsMoves` | `moves(::Neighbourhood, ::Solution)` | `Any` (iterable of `Move`) |
| `SupportsObjectiveValue` | `objective_value(::Solution)` | `Union{Nothing, <:Real}` |
| `SupportsObjectiveValueIncrement` | `objective_value_increment(::Move, ::Solution)` | `Union{Nothing, <:Real}` |
| `SupportsRandomMove` | `random_move(::Neighbourhood, ::Solution)` | `Union{Nothing, Move}` |
| `SupportsRandomMovesWithoutReplacement` | `random_moves_without_replacement(::Neighbourhood, ::Solution)` | `Any` (iterable of `Move`) |
| `SupportsRandomSolution` | `random_solution(::Problem)` | `Solution` |

### Composite abstract types

These are defined in `RoarNetAPI.Types`:

- **`Problem`** - should implement `empty_solution`, and optionally
  `construction_neighbourhood`, `local_neighbourhood`,
  `destruction_neighbourhood`, `heuristic_solution`, `random_solution`.
- **`Solution`** - should implement `copy_solution`, `objective_value`,
  and optionally `lower_bound`.
- **`Move`** - should implement `apply_move`, and optionally
  `lower_bound_increment`, `objective_value_increment`, `invert_move`.
- **`Neighbourhood`** - should implement `moves`, and optionally
  `random_moves_without_replacement`, `random_move`.

### Algorithms

All algorithms are in `RoarNetAPI.Algorithms`. Entry-point arguments are
typed against the abstract markers:

| Algorithm | Function | Signature |
|---|---|---|
| Beam search | `beam_search` | `beam_search(problem::Problem; solution::Union{Nothing, Solution}=nothing, bw::Integer=10)::Solution` |
| Best improvement | `best_improvement` | `best_improvement(problem::Problem, solution::Solution)::Solution` |
| First improvement | `first_improvement` | `first_improvement(problem::Problem, solution::Solution)::Solution` |
| GRASP | `grasp` | `grasp(problem::Problem, budget::Real; solution::Union{Nothing, Solution}=nothing, alpha::Real=0.1, local_search::Union{Nothing, Function}=nothing)::Solution` |
| Greedy construction | `greedy_construction` | `greedy_construction(problem::Problem; solution::Union{Nothing, Solution}=nothing)::Solution` |
| Random local search | `rls` | `rls(problem::Problem, solution::Solution, budget::Real)::Solution` |
| Simulated annealing | `sa` | `sa(problem::Problem, solution::Solution, budget::Real, init_temp::Real; temperature::Union{Nothing, Function}=nothing, acceptance::Union{Nothing, Function}=nothing)::Solution` |

## Typing conventions

The library uses strong typing throughout:

- **Abstract markers** (`Problem`, `Solution`, `Move`, `Neighbourhood`)
  are the dispatch keys for interface functions. Every algorithm
  argument that is a "model" object is declared against these markers.
- **`Optional[T]`** in the Python spec maps to `Union{Nothing, T}` in
  Julia. Numeric returns that may be infeasible use
  `Union{Nothing, <:Real}`.
- **Iterable returns** (`moves`, `random_moves_without_replacement`)
  are declared `::Any` because concrete implementations may return
  either a `Vector{<:Move}` or a `Channel{<:Move}`. The contract is
  "an iterable of `Move`", enforced by docstring and the test suite.
- **Internal helpers** that pair a move with a numeric increment use
  the typed accumulator `Vector{Tuple{Any, <:Real}}` so the
  heterogeneous move column is explicit and the increment column is
  restricted to real numbers.
- **Callable hooks** (`temperature`, `acceptance`, `local_search`)
  are typed as `::Union{Nothing, Function}` so users can pass any
  Julia callable or a functor struct.
- **Concrete implementations** should add explicit return types on
  every method (e.g. `objective_value(sol::MySolution)::Int`), so
  that `hasmethod` tests catch regressions and the dispatch chain
  is type-stable.

## Development

### Julia version

This library targets Julia 1.9+.

### Running the TSP example

```bash
julia --project=julia julia/examples/tsp/tsp.jl < instances/test.tsp
```

### Running tests

```bash
julia --project=julia -e 'using Pkg; Pkg.test()'
```

### Code style

The code follows standard Julia style conventions (see the
[Julia style guide](https://docs.julialang.org/en/v1/manual/style-guide/)).

## Conversion architecture

This Julia port mirrors the Python prototype-based architecture using
Julia idioms:

| Python | Julia |
|---|---|
| `Protocol` class with `__init__` typing | Abstract type + typed generic function (e.g. `apply_move(::Move, ::Solution)::Solution`) |
| `TypeVar("T", bound=X)` | `T <: X` type parameter |
| `@final` class | `struct` |
| `self` parameter | First argument (Julia convention) |
| `Optional[T]` | `Union{Nothing, T}` |
| `Union[int, float]` | `Union{Int, <:AbstractFloat}` |
| `Iterable[T]` generator | `Channel` or `Vector` |
| `logging.getLogger` | `@info` / `@debug` macros |
| `random.randrange(n)` | `rand(0:n-1)` |
| `random.shuffle(list)` | `shuffle!(vector)` |
| `random.choice(list)` | `rand(list)` |
| `random.random()` | `rand()` |
| `bisect` module | `searchsortedlast` |
| `math.isqrt` | `isqrt` |
| `perf_counter()` | `time()` |
| `TextIO` | `IO` |
| `dict[int,int]` | `Dict{Int,Int}` |
| `@classmethod` | Constructor function on the type |

## Copyright and license

Copyright and licence information is declared for each file using the
REUSE Specification Version 3.3. Use of any material must comply with
its licence.

## Acknowledgments

This project is based upon work from COST Action [Randomised
Optimisation Algorithms Research Network
(ROAR-NET)](https://www.roar-net.eu/), CA22137, supported by COST
(European Cooperation in Science and Technology).

COST ([European Cooperation in Science and
Technology](https://www.cost.eu)) is a funding agency for research and
innovation networks. Our Actions help connect research initiatives
across Europe and enable scientists to grow their ideas by sharing
them with their peers. This boosts their research, career and
progression.
