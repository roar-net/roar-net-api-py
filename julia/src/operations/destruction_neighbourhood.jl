"""
    destruction_neighbourhood(problem::Problem) -> neighbourhood::Neighbourhood

Return the destruction neighbourhood for `problem`, used by destruction-based
algorithms such as iterated local search or large neighbourhood search.
"""
function destruction_neighbourhood(::Problem)::Neighbourhood end
