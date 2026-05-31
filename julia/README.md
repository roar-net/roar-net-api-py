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
(and re-exported). Each function defines a contract that concrete types
must implement:

| Operation | Julia function | Purpose |
|---|---|---|
| `SupportsApplyMove` | `apply_move(move, solution)` | Apply a move to a solution |
| `SupportsConstructionNeighbourhood` | `construction_neighbourhood(problem)` | Get constructive neighbourhood |
| `SupportsCopySolution` | `copy_solution(solution)` | Copy a solution |
| `SupportsDestructionNeighbourhood` | `destruction_neighbourhood(problem)` | Get destruction neighbourhood |
| `SupportsEmptySolution` | `empty_solution(problem)` | Create empty solution |
| `SupportsHeuristicSolution` | `heuristic_solution(problem)` | Create heuristic solution |
| `SupportsInvertMove` | `invert_move(move)` | Invert a move |
| `SupportsLocalNeighbourhood` | `local_neighbourhood(problem)` | Get local neighbourhood |
| `SupportsLowerBound` | `lower_bound(solution)` | Get lower bound |
| `SupportsLowerBoundIncrement` | `lower_bound_increment(move, solution)` | Lower bound increment of a move |
| `SupportsMoves` | `moves(neighbourhood, solution)` | List all moves |
| `SupportsObjectiveValue` | `objective_value(solution)` | Get objective value |
| `SupportsObjectiveValueIncrement` | `objective_value_increment(move, solution)` | Objective increment of a move |
| `SupportsRandomMove` | `random_move(neighbourhood, solution)` | Get a random move |
| `SupportsRandomMovesWithoutReplacement` | `random_moves_without_replacement(neighbourhood, solution)` | Random moves without replacement |
| `SupportsRandomSolution` | `random_solution(problem)` | Create random solution |

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

All algorithms are in `RoarNetAPI.Algorithms`:

| Algorithm | Function | Requires |
|---|---|---|
| Beam search | `beam_search(problem; bw=10)` | `construction_neighbourhood`, `moves`, `apply_move`, `lower_bound_increment`, `lower_bound`, `copy_solution` |
| Best improvement | `best_improvement(problem, solution)` | `local_neighbourhood`, `moves`, `objective_value_increment`, `apply_move` |
| First improvement | `first_improvement(problem, solution)` | `local_neighbourhood`, `random_moves_without_replacement`, `objective_value_increment`, `apply_move` |
| GRASP | `grasp(problem, budget; alpha=0.1)` | `construction_neighbourhood`, `moves`, `lower_bound_increment`, `copy_solution`, `apply_move` |
| Greedy construction | `greedy_construction(problem)` | `construction_neighbourhood`, `moves`, `lower_bound_increment`, `apply_move` |
| Random local search | `rls(problem, solution, budget)` | `local_neighbourhood`, `random_moves_without_replacement`, `objective_value_increment`, `apply_move` |
| Simulated annealing | `sa(problem, solution, budget, init_temp)` | `local_neighbourhood`, `random_moves_without_replacement`, `objective_value_increment`, `apply_move`, `copy_solution` |

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
| `Protocol` class with `__init__` typing | Abstract type + generic function |
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
