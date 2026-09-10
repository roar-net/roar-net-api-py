
# SPDX-FileCopyrightText: © 2025 Authors of the roar-net-api-py project <https://github.com/roar-net/roar-net-api-py/blob/main/AUTHORS>
#
# SPDX-License-Identifier: Apache-2.0
"""
    Neighbourhood

Abstract type representing a neighbourhood structure.
A neighbourhood should implement `moves`, and optionally
`random_moves_without_replacement` and `random_move`.
"""
abstract type Neighbourhood end
