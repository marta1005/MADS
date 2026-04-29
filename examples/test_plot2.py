import os, sys

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")),
)

import numpy as np
import matplotlib.path as mpath
from postprocessing.plotVariables import plotScattersAndHistoriograms
from postprocessing.dataClassification import objectiveClassification, valuesClassification

rawData = np.genfromtxt('pareto_data-eng.txt',skip_header=1,delimiter=',')

nPoints = len(rawData)

SW = 33.9
rho = 0.549526
v = 154.835

cd_0 = 0.023

MTOW = 32000.0
OEW_W = 24000.0

g = 9.81

L_i = MTOW*g

D_0 = rho*v*v*SW*cd_0
cTip_Min = 0.2

def caseAnalysis(vector):

    if vector[0] > 0.0:
        valid = True
    else:
        valid = False

    D_i = L_i/vector[5] + D_0
    eff_i = L_i/D_i

    wingMass = vector[7]

    L_f = (OEW_W+wingMass)*g
    D_f = L_f/vector[6] + D_0
    eff_f = L_f/D_f

    Range = vector[8]/(vector[5]+vector[6])*(eff_i+eff_f)

    span = vector[1] + vector[2]

    pKink = vector[1]/span

    c_Min = SW/span
    c_Max = (SW-vector[2]*cTip_Min/2.0)/(vector[1]+vector[2]/2.0)

    pChord = (vector[3]-c_Min)/(c_Max-c_Min)

    return(valid,[eff_i,eff_f,wingMass,span,pKink,pChord,Range])

data = []
valid = []

for case in rawData:
    
    valid_case, data_case = caseAnalysis(case)
    valid.append(valid_case)
    data.append(data_case)

data = np.array(data)

minRange,maxRange,rangeCut,objectives = objectiveClassification(data[:,6],valid)

values = valuesClassification([data[:,3],data[:,4],data[:,5]],data[:,6],minRange,maxRange,rangeCut,valid)
values2 = valuesClassification([data[:,2],data[:,0]],data[:,6],minRange,maxRange,rangeCut,valid)

bins = [6, 9, 6]

names = ['span', 'kink', 'chord_root']

markers = [None for x in range(4)]

sector1 = mpath.Path.arc(0.0,90.0, is_wedge=True)
sector2 = mpath.Path.arc(180.0,270.0, is_wedge=True)
circle = mpath.Path.circle()
# concatenate the circle with an internal cutout of the sectors
centroid = mpath.Path(
    vertices=np.concatenate([circle.vertices, sector1.vertices[::-1, ...], sector2.vertices[::-1, ...]]),
    codes=np.concatenate([circle.codes, sector1.codes, sector2.codes]))


markers[0] = dict(label = 'Not Valid', marker="X",  s=140, color='red', alpha = 1.0)
markers[1] = dict(label = 'Valid', marker="o",  s=120, color='blue', alpha = 0.2)
markers[2] = dict(label = 'Close To Optimal', marker="o",  s=100, color='green', alpha = 0.6)
markers[3] = dict(label = 'Optimal', marker=centroid,  s=80 , color='magenta', edgecolor='darkmagenta', alpha = 1.0)

#plotScattersAndHistoriograms(values,names,markers,bins,objectives,'range',True,False)

names2 = ['wing_Mass', 'efficiency']
bins2 = [10, 10]

plotScattersAndHistoriograms(values2,names2,markers,bins2,objectives,'range',False,True)

minMass,maxMass,massCut,mObjectives = objectiveClassification(data[:,2],valid,False)
values3 = valuesClassification([data[:,3],data[:,4],data[:,5]],data[:,2],minMass,maxMass,massCut,valid,False)
#plotScattersAndHistoriograms(values3,names,markers,bins,mObjectives,'wing_mass',True,False,False)

minEff,maxEff,effCut,eObjectives = objectiveClassification(data[:,0],valid)
values4 = valuesClassification([data[:,3],data[:,4],data[:,5]],data[:,0],minEff,maxEff,effCut,valid)
#plotScattersAndHistoriograms(values4,names,markers,bins,eObjectives,'efficiency',True,False)
