# SPDX-FileCopyrightText: © 2025 Authors of the roar-net-api-py project <https://github.com/roar-net/roar-net-api-py/blob/main/AUTHORS>
#
# SPDX-License-Identifier: Apache-2.0
"""
    local_neighbourhood(problem::Problem) -> neighbourhood::Neighbourhood

Return the local (improvement) neighbourhood for `problem`, used by local
search algorithms such as best improvement, first improvement, RLS, and SA.
"""
function local_neighbourhood(::Problem)::Neighbourhood end
