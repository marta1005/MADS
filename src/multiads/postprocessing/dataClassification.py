import numpy as np
from numpy.typing import NDArray


def objectiveClassification(objective,validity,maximize=True):

    firstValue = True

    for iValue in range(len(objective)):
        if (validity[iValue]):
            if firstValue:
                maxVal = objective[iValue]
                minVal = objective[iValue]
                firstValue = False
            else:
                maxVal = max(maxVal,objective[iValue])
                minVal = min(minVal,objective[iValue])

    if (maximize):
        valueCut = 0.1*minVal + 0.9*maxVal
    else:
        valueCut = 0.9*minVal + 0.1*maxVal

    objectiveArray = [[] for i in range(4) ]

    for iValue in range(len(objective)):

        if not(validity[iValue]):
            objectiveArray[0] = np.concatenate((objectiveArray[0],[objective[iValue]]))
        elif (maximize):
            if (objective[iValue] < valueCut):
                objectiveArray[1] = np.concatenate((objectiveArray[1],[objective[iValue]]))
            elif (objective[iValue] < maxVal):
                objectiveArray[2] = np.concatenate((objectiveArray[2],[objective[iValue]]))
            else:
                objectiveArray[3] = np.concatenate((objectiveArray[3],[objective[iValue]]))
        else:
            if (objective[iValue] > valueCut):
                objectiveArray[1] = np.concatenate((objectiveArray[1],[objective[iValue]]))
            elif (objective[iValue] > minVal):
                objectiveArray[2] = np.concatenate((objectiveArray[2],[objective[iValue]]))
            else:
                objectiveArray[3] = np.concatenate((objectiveArray[3],[objective[iValue]]))

    return(minVal,maxVal,valueCut,objectiveArray)

def valuesClassification(data,objective,minVal,maxVal,valueCut,validity,maximize=True):

    nVars = len(data)

    dataArray = [[[] for x in range(4)] for y in range(nVars)]

    for iValue in range(len(objective)):

        if not(validity[iValue]):
            for iVar in range(nVars):
                dataArray[iVar][0] = np.concatenate((dataArray[iVar][0],[data[iVar][iValue]]))
        elif (maximize):
            if (objective[iValue] < valueCut):
                for iVar in range(nVars):
                    dataArray[iVar][1] = np.concatenate((dataArray[iVar][1],[data[iVar][iValue]]))
            elif (objective[iValue] < maxVal):
                for iVar in range(nVars):
                    dataArray[iVar][2] = np.concatenate((dataArray[iVar][2],[data[iVar][iValue]]))
            else:
                for iVar in range(nVars):
                    dataArray[iVar][3] = np.concatenate((dataArray[iVar][3],[data[iVar][iValue]]))
        else:
            if (objective[iValue] > valueCut):
                for iVar in range(nVars):
                    dataArray[iVar][1] = np.concatenate((dataArray[iVar][1],[data[iVar][iValue]]))
            elif (objective[iValue] > minVal):
                for iVar in range(nVars):
                    dataArray[iVar][2] = np.concatenate((dataArray[iVar][2],[data[iVar][iValue]]))
            else:
                for iVar in range(nVars):
                    dataArray[iVar][3] = np.concatenate((dataArray[iVar][3],[data[iVar][iValue]]))

    return(dataArray)



            
