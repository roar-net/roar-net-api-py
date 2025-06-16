<!--
SPDX-FileCopyrightText: © 2025 Authors of the roar-net-api-py project <https://github.com/roar-net/roar-net-api-py/blob/main/AUTHORS>

SPDX-License-Identifier: CC-BY-4.0
-->

# ROAR-NET API - Python

This repository contains a library with the python implementation of
the ROAR-NET API Specification and several algorithms. You can find
the full specification in the
[roar-net-api-spec](https://github.com/roar-net/roar-net-api-spec)
repository.

## Contents

This library contains protocol types for all operations in the
specification, and several algorithms that are currently supported by
the specification.

### Protocol types

Protocol types for the operations in the ROAR-NET API Specification
are defined in the module
[roar_net_api.operations](https://github.com/roar-net/roar-net-api-py/tree/main/src/roar_net_api/operations).
For example, the operation
[apply_move](https://github.com/roar-net/roar-net-api-py/tree/main/src/roar_net_api/operations/apply_move.py)
is defined in a protocol `SupportsApplyMove`.

This library considers an object-oriented approach, such that
operations should be implemented by a class of the first argument
type. For example, for the operation:

```text
apply_move(Move, Solution) -> Solution
```

we consider that it should be implemented as part of the class `Move`,
which means that the protocol type is defined as:

```python
SolutionT = TypeVar("SolutionT")

class SupportsApplyMove(Protocol[SolutionT]):
    def apply_move(self, solution: SolutionT) -> SolutionT: ...
```

### Algorithms

We support several algorithms that can be implemented following the
ROAR-NET API Specification. You can find the algorithms in the module
[roar_net_api.algorithms](https://github.com/roar-net/roar-net-api-py/tree/main/src/roar_net_api/algorithms). Here
is the current list of supported algorithms:

- Beam search: `beam_search`
- Best improvement: `best_improvement`
- First improvement: `first_improvement`
- GRASP: `grasp`
- Greedy construction: `greedy_construction`
- Random local search: `rls`
- Simulated annealing: `sa`

## Using

### Adding it to your project

To use this library add it to your project. For example, to add it to
a `uv` project you can do:

```bash
uv add git+https://github.com/roar-net/roar-net-api-py --branch training-school
```

Alternatively you can add it to a `requirements.txt` file:

```text
roar-net-api @ git+https://github.com/roar-net/roar-net-api-py.git@training-school
```

Or install it directly with `pip`:

```bash
pip install "git+https://github.com/roar-net/roar-net-api-py.git@training-school"
```

### Implement a model

To implement a model, we recommend that you take advantage of python
type hints to warn you about potential issues. For example, if you
would like your problem class to support the operations
`empty_solution` and `constructive_neighbourhood`, you can inherit
from the protocol types for these operations to get type checking, for
example:

```python
from roar_net_api.operations import (
  SupportsEmptySolution,
  SupportsConstructiveNeighbourhood
)

class ConstructiveNeighbourhood:
  ...

class Solution:
  ...

class Problem(
  SupportsEmptySolution[Solution],
  SupportsConstructiveNeighbourhood[ConstructiveNeighbourhood]
):
  def empty_solution(self) -> Solution:
    ...

  def constructive_neighbourhood(self) -> ConstructiveNeighbourhood:
    ...
```

where `...` would be for your implementation.

If you do this, your editor/IDE can inform you of potential issues in
the implementation.

For a full example, see the
[tsp.py](https://github.com/roar-net/roar-net-api-py/blob/main/examples/tsp/tsp.py)
file in the examples folder, which implements a model for the
travelling salesman problem that can be solved by all algorithms.

## Development

This library targets Python 3.11, in order to support the latest
version of `pypy` which allows for faster execution of algorithms. As
such, make sure to test your changes with the latest version of
`pypy`.

The project is managed with `uv`, please follow the instructions on
their [website](https://docs.astral.sh/uv/) to install it.

The code is checked with the following linters:

- `ruff`
- `mypy` in `strict` mode

Pull requests are checked against these linters.

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
innovation.

<br/>
<img
  src="https://raw.githubusercontent.com/roar-net/.github/refs/heads/main/images/costeu.png"
  alt="COST and European Union Logos"
  width=460px
/>
