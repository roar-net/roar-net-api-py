"""
    local_neighbourhood(problem::Problem) -> neighbourhood::Neighbourhood

Return the local (improvement) neighbourhood for `problem`, used by local
search algorithms such as best improvement, first improvement, RLS, and SA.
"""
function local_neighbourhood(::Problem)::Neighbourhood end
