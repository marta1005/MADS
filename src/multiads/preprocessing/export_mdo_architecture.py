import gemseo


# plot xsdm matrix
def export_scenario(mads_scenario):
    mads_scenario.scenario.xdsmize(show_html=True)


# plot n2 matrix
def export_n2_matrix(disciplines):
    gemseo.generate_n2_plot(disciplines, save=True, show=False)
