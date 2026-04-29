import numpy as np
from numpy.typing import NDArray
import matplotlib.path as mpath
from postprocessing.plotVariables import plotScattersAndHistoriograms
from postprocessing.dataClassification import objectiveClassification, valuesClassification


def optimizationResults(scenario,maximizeObservable=None,save=True,show=False):

    markers = []
    sector1 = mpath.Path.arc(0.0,90.0, is_wedge=True)
    sector2 = mpath.Path.arc(180.0,270.0, is_wedge=True)
    circle = mpath.Path.circle()
    # concatenate the circle with an internal cutout of the sectors
    centroid = mpath.Path(
        vertices=np.concatenate([circle.vertices, sector1.vertices[::-1, ...], sector2.vertices[::-1, ...]]),
        codes=np.concatenate([circle.codes, sector1.codes, sector2.codes]))
    markers.append(dict(label = 'Not Valid', marker="X",  s=140, color='red', alpha = 1.0))
    markers.append(dict(label = 'Valid', marker="o",  s=120, color='blue', alpha = 0.2))
    markers.append(dict(label = 'Close To Optimal', marker="o",  s=100, color='green', alpha = 0.6))
    markers.append(dict(label = 'Optimal', marker=centroid,  s=80 , color='magenta', edgecolor='darkmagenta', alpha = 1.0))


    objectiveVar = str(scenario.formulation.opt_problem.objective)
    observableVars = scenario.formulation.opt_problem.observables
    
    feasible = scenario.formulation.opt_problem.get_feasible_points()[0]
    points = scenario.formulation.opt_problem.database.get_x_vect_history()
    valid = []
    for x in points:
        if np.any(feasible  == x):
            valid.append(True)
        else:
            valid.append(False)
    points = np.array(points)

    pointVars = scenario.formulation.opt_problem.design_space.variable_names
    pointNames = []
    pointBins = []
    for var in pointVars:
        pointNames.append(str(var))
        pointBins.append(10)

    if (objectiveVar[0:1] == '-'):
        maximize = True
        objectiveName = objectiveVar[1:].split('(')[0]
    else:
        maximize = False
        objectiveName = objectiveVar.split('(')[0]
    objectiveVar = objectiveVar.split('(')[0]
    
    objectiveValues = np.array(list(scenario.formulation.opt_problem.get_data_by_names(objectiveVar).values())[0])
    if maximize:
        objectiveValues = -objectiveValues

    observableNames = []
    observableValues = []
    observableBins = []
    observableMax = []
    for i,var in enumerate(observableVars):
        name = str(var).split('(')[0]
        observableNames.append(name)
        values = np.array(list(scenario.formulation.opt_problem.get_data_by_names(name).values())[0])
        observableValues.append(values)
        observableBins.append(10)
        if not(maximizeObservable == None):
            observableMax.append(maximizeObservable[i])
        else:
            observableMax.append(True)

    minObj,maxObj,cutObj,objMatrix = objectiveClassification(
        objectiveValues,
        valid,
        maximize)

    pointMatrix = valuesClassification(
        points.transpose(),
        objectiveValues,
        minObj,
        maxObj,
        cutObj,
        valid)
    plotScattersAndHistoriograms(
        pointMatrix,
        pointNames,
        markers,
        pointBins,
        objMatrix,
        objectiveName,
        save,
        show,
        maximize)

    if (len(observableNames)>1):

        obsMatrix = valuesClassification(
            observableValues,
            objectiveValues,
            minObj,
            maxObj,
            cutObj,
            valid)
        plotScattersAndHistoriograms(
            obsMatrix,
            observableNames,
            markers,
            observableBins,
            objMatrix,
            objectiveName,
            save,
            show,
            maximize)

        for i in range(len(observableNames)):
            minObs,maxObs,cutObs,obsMatrix = objectiveClassification(
                observableValues[i],
                valid,
                observableMax[i])
            pointMatrix = valuesClassification(
                points.transpose(),
                observableValues[i],
                minObs,
                maxObs,
                cutObs,
                valid,
                observableMax[i])
            plotScattersAndHistoriograms(
                pointMatrix,
                pointNames,
                markers,
                pointBins,
                obsMatrix,
                observableNames[i],
                save,
                show,
                observableMax[i])





