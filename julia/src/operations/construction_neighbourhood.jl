"""
    construction_neighbourhood(problem::Problem) -> neighbourhood::Neighbourhood

Return the constructive neighbourhood for `problem`, used by constructive
algorithms such as greedy construction, beam search, and GRASP.
"""
function construction_neighbourhood(::Problem)::Neighbourhood end
