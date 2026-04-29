import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from numpy.typing import NDArray
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import Delaunay

def plotScattersAndHistoriograms(
    variablesValues,
    variablesNames,
    dataMarkers,
    variablesBins,
    objectivesValues,
    objectiveName,
    save,
    show,
    maximize=True):

    sizes = [len(variablesValues), len(variablesValues[0])]
    fig = plt.figure(figsize=(16, 9), dpi=80)
    
    if maximize:
        title = str("max[ ")
        file = str("max_")
    else:
        title = str("min[ ")
        file = str("min_")

    title = title + str(objectiveName) + str("(")
    file = file + str(objectiveName) + str("_")

    for i in range(sizes[0]):
        title = title + str(variablesNames[i])
        file = file + str(variablesNames[i])
        if (i == sizes[0]-1):
            title = title + str(") ]")
            file = file + str(".png")
        else:
            title = title + str(",")
            file = file + str(".")

    titleTex = fig.suptitle(title, fontsize=16)

    jointVar = [[] for x in range(sizes[0])]
    jointVarC = [[] for x in range(sizes[0])]

    axs = [ [None for x in range(sizes[0])] for y in range(sizes[0])]

    jointObj = []

    initC = 1

    cmap=cm.jet

    for i in reversed(range(initC,sizes[1])):
        if (len(objectivesValues[i])>0):
            sortOrder = np.argsort(objectivesValues[i])
            if not(maximize):
                sortOrder = list(reversed(sortOrder))
            jointObj = np.concatenate((jointObj,objectivesValues[i][sortOrder]))
        
            for j in range(sizes[0]):
                jointVar[j] = np.concatenate((jointVar[j],variablesValues[j][i][sortOrder]))
                if (i>=initC):
                    jointVarC[j] = np.concatenate((jointVarC[j],variablesValues[j][i][sortOrder]))

    factorLine = 0.1
    dz = factorLine*(max(jointObj)-min(jointObj))
    z0 = min(jointObj) - dz
    zf = max(jointObj) + dz
    hCut = (zf-z0)/10.0

    for i in range(sizes[0]):

        for j in range(sizes[0]):

            iFig = sizes[0]*i+j+1

            if (i == j):
                
                axs[i][j] = fig.add_subplot(sizes[0],sizes[0],iFig)
                axs[i][j].hist(jointVar[i], bins=variablesBins[i], linewidth=0.5, edgecolor="white", color="lightcoral")
                axs[i][j].set_ylabel("Frequency")
                

            elif (i>j):

                axs[i][j] = fig.add_subplot(sizes[0],sizes[0],iFig)

                for k in range(sizes[1]):
                    if (k == sizes[1]-1):
                        for xP in variablesValues[j][k]:
                            axs[i][j].axvline(x=xP, color=dataMarkers[k].get('color'),linestyle ='--',zorder=-0.5)
                        for yP in variablesValues[i][k]:
                            axs[i][j].axhline(y=yP, color=dataMarkers[k].get('color'),linestyle ='--',zorder=-0.5)
                    axs[i][j].scatter(variablesValues[j][k], variablesValues[i][k], **dataMarkers[k],zorder=k) #s=dataMarkers[k][0], label=dataMarkers[k][1], color=dataMarkers[k][2], marker=dataMarkers[k][3],alpha=dataMarkers[k][4])
                    
                axs[i][j].set_ylabel(variablesNames[i])
                axs[i][j].legend(loc='upper right')

            
            else:

                axs[i][j] = fig.add_subplot(sizes[0],sizes[0],iFig,projection='3d')
                #tri = Delaunay(np.array([jointVarC[j],jointVarC[i]]).T)
                #cntr = axs[i][j].voxels(jointVarC[j], jointVarC[i], jointObj,True, None)#cmap=cm.coolwarm,alpha=0.5)
                points = []
                for k in range(len(jointVarC[j])):
                    points.append([jointVarC[j][k], jointVarC[i][k], jointObj[k]])
                
                points = np.array(points)

                # Hago la triangulacion de delaunay
                tri = Delaunay(points)
                # Cojo las 4 caras triangulares de los tetraedros
                tri2D = []
                for elem in tri.simplices:
                    tri2D.append([elem[0],elem[1],elem[2]]);
                    tri2D.append([elem[0],elem[1],elem[3]]);
                    tri2D.append([elem[0],elem[2],elem[3]]);
                    tri2D.append([elem[1],elem[2],elem[3]]);

                triangleList = CreateTriangleList(points,hCut)

                PlotTriangleList(triangleList,axs[i][j],cmap,z0,zf,maximize)

                #axs[i][j].plot_trisurf(points[:,0], points[:,1], points[:,2], triangles=tri2D, linewidth = 2.0, cmap='jet', alpha=0.5)
                
                #axs[i][j].plot_trisurf(jointVarC[j], jointVarC[i], jointObj, cmap=cm.coolwarm,alpha=0.5,zorder=-0.5)
                dx = factorLine*(max(jointVar[j])-min(jointVar[j]))
                x0 = min(jointVar[j]) - dx
                xf = max(jointVar[j]) + dx

                dy = factorLine*(max(jointVar[i])-min(jointVar[i]))
                y0 = min(jointVar[i]) - dy
                yf = max(jointVar[i]) + dy

                for k in range(2,sizes[1]):
                    if (k == sizes[1]-1):
                        
                        for l in range(len(variablesValues[i][k])):
                        
                            axs[i][j].plot(
                                [x0,xf],
                                [variablesValues[i][k][l],variablesValues[i][k][l]],
                                [z0,z0],
                                color=dataMarkers[k].get('color'),linestyle ='--',zorder=-1.0)
                            axs[i][j].plot(
                                [variablesValues[j][k][l],variablesValues[j][k][l]],
                                [y0,yf],
                                [z0,z0],
                                color=dataMarkers[k].get('color'),linestyle ='--',zorder=-1.0)
                            axs[i][j].plot(
                                [variablesValues[j][k][l],variablesValues[j][k][l]],
                                [variablesValues[i][k][l],variablesValues[i][k][l]],
                                [z0,objectivesValues[k][l]],
                                color=dataMarkers[k].get('color'),linestyle ='--',zorder=-1.0)
                        
                        #axs[i][j].stem(variablesValues[j][k], variablesValues[i][k], objectivesValues[k],linefmt ='m--')
                    axs[i][j].scatter(variablesValues[j][k], variablesValues[i][k], objectivesValues[k],**dataMarkers[k],zorder=k)
                axs[i][j].set_ylabel(variablesNames[i])
                axs[i][j].set_zlabel(objectiveName)
                axs[i][j].set_xlim([x0,xf])
                axs[i][j].set_ylim([y0,yf])
                axs[i][j].set_zlim([z0,zf])

                
                
            axs[i][j].set_xlabel(variablesNames[j])
            axs[i][j].grid(True)

    box = axs[0][0]._position

    width = box.x1- box.x0
    height = box.y1 - box.y0

    factor = 16.0/9.0

    for i in range(sizes[0]):
        for j in range(sizes[0]):
            if (j>i):
                boxT = axs[i][j]._position
                xT = (boxT.x0 + boxT.x1)/2.0
                yT = (boxT.y0 + boxT.y1)/2.0
                axs[i][j].set_position([xT-width*factor/2.0,yT-height*factor/2.0,factor*width,factor*height])
                axs[i][j].set_box_aspect((2.0*factor*width, 2.0*factor*width, height))
                axs[i][j].set_facecolor((0.0,0.0,0.0,0.0))

    
    if save:

        figH = 4.5*sizes[0]
        figW = 8.0*sizes[0]
        titleTex.set_fontsize(max(5*sizes[0],20))

        fig.set_size_inches(figW,figH)
        plt.savefig(file,dpi=80.0,format="png")

    if show:
        titleTex.set_fontsize(15)
        fig.set_size_inches(24,13.5)
        plt.show()

def CreateTriangleList(points,hCut):

    hull = ConvexHull(points,qhull_options="Qs")
    triangleList = []
                
    new = True
    for simplex in hull.simplices:
        tTemp = triangle(points[simplex[0],:],points[simplex[1],:],points[simplex[2],:])
        if (tTemp.area > 0.0):
            if (new):
                hMax = tTemp.maxHeight
                new = False
            else:
                hMax = max(hMax,tTemp.maxHeight)
            triangleList.append(tTemp)
                
    while hMax>hCut:
        new = True
        triangleList2 = []
        for t in triangleList:
            if (t.area > 0.0):
                if (t.maxHeight < hCut):
                    triangleList2.append(t)
                    if (new):
                        hMax = t.maxHeight
                        new = False
                    else:
                        hMax = max(hMax,t.maxHeight)
                else:
                    tList = t.exploitTriangle
                    for tTemp in tList:
                        triangleList2.append(tTemp)
                        if (new):
                            hMax = tTemp.maxHeight
                            new = False
                        else:
                            hMax = max(hMax,tTemp.maxHeight)
        triangleList = triangleList2

    return(triangleList)

def PlotTriangleList(triangleList,axle,cmap,z0,zf,maximize):
    for t in triangleList:
        z = t.getPoint(1.0/3.0,1.0/3.0)
        zc = (z[2]-z0)/(zf-z0)
        if not(maximize):
            zc = 1.0 - zc
        c = cmap(zc)
        t.plot(axle,[c[0],c[1],c[2],0.9])

class triangle:
    def __init__(self,p1,p2,p3):
        self.p1 = np.array(p1)
        self.p2 = np.array(p2)
        self.p3 = np.array(p3)

        p12 = np.subtract(self.p2,self.p1)
        self.u = p12 / np.linalg.norm(p12)
        p13 = np.subtract(self.p3,self.p1)
        self.d = np.dot(p13,self.u)
        v = p13 - self.d*self.u
        self.h = np.linalg.norm(v)
        self.v = v / self.h
        self.w = np.cross(self.u,self.v)

        self.l = np.array([0.0, 0.0, 0.0])

        self.l[0] = np.linalg.norm(np.subtract(self.p1,self.p2))
        self.l[1] = np.linalg.norm(np.subtract(self.p2,self.p3))
        self.l[2] = np.linalg.norm(np.subtract(self.p3,self.p1))

        self.z = np.array([0.0, 0.0, 0.0])
        self.z[0] = abs(np.dot((np.subtract(self.p1,self.p2)),np.array([0.0,0.0,1.0])))
        self.z[1] = abs(np.dot((np.subtract(self.p2,self.p3)),np.array([0.0,0.0,1.0])))
        self.z[2] = abs(np.dot((np.subtract(self.p3,self.p1)),np.array([0.0,0.0,1.0])))

        self.area = self.h*self.l[0]/2.0

    @property
    def maxLenght(self):

        return(max(self.l))

    @property
    def maxHeight(self):

        return(max(self.z))

    @property
    def exploitTriangle(self):

        p4 = (self.p1+self.p2)/2.0
        p5 = (self.p2+self.p3)/2.0
        p6 = (self.p3+self.p1)/2.0

        return([triangle(p6,self.p1,p4),triangle(p4,self.p2,p5),triangle(p5,self.p3,p6),triangle(p4,p5,p6)])
        
    def calcUV(self,p):

        a = np.array(p)-self.p1
        b = a - np.dot(a,self.w)*self.w

        v = np.dot(b,self.v)/self.h
        u = (np.dot(b,self.u)-self.d*v)/self.l[0]

        return[u,v]

    @classmethod
    def interpol(self,f1,f2,f3,u,v):
        return(f1*(1.0-u)*(1-v) + f2*u + f3*v)

    def getPoint(self,u,v):
        return(self.p1*(1.0-u)*(1-v) + self.p2*u + self.p3*v)

    def plot(self,axle,color):
        triangles = [(  (self.p1[0],self.p1[1], self.p1[2]),
                        (self.p2[0],self.p2[1], self.p2[2]),
                        (self.p3[0],self.p3[1], self.p3[2]))]
                        
        axle.add_collection3d(Poly3DCollection(triangles,edgecolor = 'k', facecolor=color, linewidth = 1.0))




    