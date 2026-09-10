# SPDX-FileCopyrightText: © 2025 Authors of the roar-net-api-py project <https://github.com/roar-net/roar-net-api-py/blob/main/AUTHORS>
#
# SPDX-License-Identifier: Apache-2.0
"""
    destruction_neighbourhood(problem::Problem) -> neighbourhood::Neighbourhood

Return the destruction neighbourhood for `problem`, used by destruction-based
algorithms such as iterated local search or large neighbourhood search.
"""
function destruction_neighbourhood(::Problem)::Neighbourhood end
