


############################################################################################
# Set Environment
from gemseo.utils.study_analyses.mdo_study_analysis import MDOStudyAnalysis
import gemseo

# local settings - only for development
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..', '../../Tools/multiads/src')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..', '../../Tools/multiads/src/scenario')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..', '../../Tools/multiads/src/discipline')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..', '../../Tools/multiads/src/wrapper')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..', '../../Tools/multiads/src/connectors')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..', '../../../Tools/DUST_interface/')))


#############################################################################################

# generate from batch
# syst_arch = MDOStudyAnalysis("./Scenario_ODE4HERA.xlsx")
#syst_arch.generate_xdsm()
# gemseo-study -x -o out Scenario_ODE4HERA.xlsx

def architecture_analysis(mads_scenario, discipline):
    
    mads_scenario.scenario.xdsmsize(show_html=False)

    gemseo.generate_n2_plot(discipline,save=True,show=False)

    return
