# SPDX-FileCopyrightText: © 2025 Authors of the roar-net-api-py project <https://github.com/roar-net/roar-net-api-py/blob/main/AUTHORS>
#
# SPDX-License-Identifier: Apache-2.0
"""
    construction_neighbourhood(problem::Problem) -> neighbourhood::Neighbourhood

Return the constructive neighbourhood for `problem`, used by constructive
algorithms such as greedy construction, beam search, and GRASP.
"""
function construction_neighbourhood(::Problem)::Neighbourhood end
