# SPDX-FileCopyrightText: © 2025 Authors of the roar-net-api-py project <https://github.com/roar-net/roar-net-api-py/blob/main/AUTHORS>
#
# SPDX-License-Identifier: Apache-2.0
"""
    Solution

Abstract type representing a solution to an optimisation problem.
A solution should implement `copy_solution`, `objective_value`,
and optionally `lower_bound`.
"""
abstract type Solution end
