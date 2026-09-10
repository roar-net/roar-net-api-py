
# SPDX-FileCopyrightText: © 2025 Authors of the roar-net-api-py project <https://github.com/roar-net/roar-net-api-py/blob/main/AUTHORS>
#
# SPDX-License-Identifier: Apache-2.0
"""
    Problem

Abstract type representing an optimisation problem.
A problem should implement `empty_solution`, and optionally
`construction_neighbourhood`, `local_neighbourhood`, `destruction_neighbourhood`,
`heuristic_solution`, and `random_solution`.
"""
abstract type Problem end
