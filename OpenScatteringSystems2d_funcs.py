from ngsolve import *
from netgen.geom2d import SplineGeometry
import numpy as np
import scipy as sp
from scipy import io
import scipy.linalg
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.path import Path
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
import os
import time

def CreateGeometry(params, scatShape='circ', scatPos=np.array([[],[]]), scatRad=np.array([]), scatNr=1.0, scatNrLast=None, scatNi=0.0, scatNiLast=None, 
	uniformNr=1.0, uniformNi=0.0, shiftTrans=0.0, shiftTransLast=0.0, shiftLong=0.0, shiftLongLast=0.0, shiftRad=0.0, shiftRadLast=0.0, shiftAng=0.0, 
	shiftAngLast=0.0, shiftNr=0.0, shiftNrLast=0.0, shiftNiLast=0.0, shiftNi=0.0, numLeads=2, pmlType='cartesian', pmlLength=None, rectWidth2=None, polyPts=None, 
	polyCodes=None, polyDir='ccw', mazeLines=None, randNr=None, embedInd=None, msgOutput=True, meshScatRefine=True, meshEdgeRefine=True, meshCurve=True):

    """
    Create a scattering geometry consisting of a cirular scattering region bounded by a perfectly matched layer (PML) region to absorb all outgoing waves
    """

    # Convert required parameter dictionary entries to local variables
    nphwl = params['nphwl']
    k = params['k']
    feOrder = params['feOrder']

    numScat = len(scatPos[0,:])

    if isinstance(scatShape,list) is False:
        scatShape = [scatShape for i in range(numScat)]
    if 'rect' in scatShape:
        if rectWidth2 is None:
            rectWidth2 = scatRad
        if type(rectWidth2) is not np.ndarray:
            rectWidth2 = rectWidth2*np.ones(numScat) 

    if type(scatNr) is not np.ndarray and scatNr != 1.0:
        scatNr = scatNr*np.ones(numScat) 
    elif type(scatNr) is not np.ndarray and scatNr == 1.0:
        scatNr = np.ones(numScat) 
    if scatNrLast != None:
        scatNr[-1] = scatNrLast
            
    if type(scatNi) is not np.ndarray and scatNi != 0.0:
        scatNi = scatNi*np.ones(numScat) 
    elif type(scatNi) is not np.ndarray and scatNi == 0.0:
        scatNi = np.zeros(numScat) 
    if scatNiLast != None:
        scatNi[-1] = scatNiLast
    
    if type(shiftTrans) is not np.ndarray and shiftTrans != 0.0:
        shiftTrans = shiftTrans*np.ones(numScat) 
    elif type(shiftTrans) is not np.ndarray and shiftTrans == 0.0:
        shiftTrans = np.zeros(numScat) 
        if shiftTransLast != 0.0:
            shiftTrans[-1] = shiftTransLast
        
    if type(shiftLong) is not np.ndarray and shiftLong != 0.0:
        shiftLong = shiftLong*np.ones(numScat) 
    elif type(shiftLong) is not np.ndarray and shiftLong == 0.0:
        shiftLong = np.zeros(numScat) 
        if shiftLongLast != 0.0:
            shiftLong[-1] = shiftLongLast
        
    if type(shiftRad) is not np.ndarray and shiftRad != 0.0:
        shiftRad = shiftRad*np.ones(numScat) 
    elif type(shiftRad) is not np.ndarray and shiftRad == 0.0:
        shiftRad = np.zeros(numScat) 
        if shiftRadLast != 0.0:
            shiftRad[-1] = shiftRadLast
        
    if type(shiftAng) is not np.ndarray and shiftAng != 0.0:
        shiftAng = shiftAng*np.ones(numScat) 
    elif type(shiftAng) is not np.ndarray and shiftAng == 0.0:
        shiftAng = np.zeros(numScat) 
        if shiftAngLast != 0.0:
            shiftAng[-1] = shiftAngLast
        
    if type(shiftNr) is not np.ndarray and shiftNr != 0.0:
        shiftNr = shiftNr*np.ones(numScat) 
    elif type(shiftNr) is not np.ndarray and shiftNr == 0.0:
        shiftNr = np.zeros(numScat) 
        if shiftNrLast != 0.0:
            shiftNr[-1] = shiftNrLast
            
    if type(shiftNi) is not np.ndarray and shiftNi != 0.0:
        shiftNi = shiftNi*np.ones(numScat) 
    elif type(shiftNi) is not np.ndarray and shiftNi == 0.0:
        shiftNi = np.zeros(numScat) 
        if shiftNiLast != 0.0:
            shiftNi[-1] = shiftNiLast

    # Values used for refinement of edges in the empty geometry or of polygonal scatterers
    maxhVac = (2*np.pi/k)/2.0/nphwl
    if meshEdgeRefine is True:
        maxhEdge = maxhVac/5.0
    else:
        maxhEdge = maxhVac
    refVal = 0
    refOrder = 3
    refFactor = 0.5

    # If not given, choose length of PML as nlambda times the longest wavelength in propagation direction which gets dampened the least
    if pmlLength is None:
        nlambda = 1.5
        lambdaMax = 2.0*np.pi/k
        pmlLength = nlambda*lambdaMax
        
    if 'pmlRad' in params.keys():
        pmlRad = params['pmlRad']
    else:
        pmlRad = 0.1
        params['pmlRad'] = pmlRad

    # Set counter for polygons and put polyPts in a list in case there is only one polygon in order to be able to use the loop below
    polyCount = 0    
    numPoly = scatShape.count('poly')
    if isinstance(polyPts, list) is False:
        polyPtsAll = [polyPts for i in range(numPoly)]
    else:
        polyPtsAll = polyPts
    if isinstance(polyCodes, list) is False:
        polyCodesAll = [polyCodes for i in range(numPoly)]
    else:
        polyCodesAll = polyCodes
    if isinstance(polyDir, list) is False:
        polyDirAll = [polyDir for i in range(numPoly)]
    else:
        polyDirAll = polyDir

    # Define edge points for waveguide boundary and input and output PML
    geo = SplineGeometry()
    
    geo.AddCircle((0, 0), pmlRad,  leftdomain=1, rightdomain=2, bc="input")
    geo.AddCircle((0, 0), pmlRad+pmlLength,  leftdomain=2, rightdomain=0, bc="wall")
           
    # Add randomly placed scatterers
    if numScat > 0:
        
        # Create domain numbers for different scatterers using their refractive index (including the shifts). Since np.unique
        # returns a sorted array, the metallic scatterers with index > 1e3 are the last and will be given domain 0
        useDirichlet = 1e3
        scatNrefrAll = scatNr+shiftNr + 1.0j*(scatNi+shiftNi)
        scatNrefrUnique = np.unique(scatNrefrAll)
        scatDom = np.zeros(numScat)
        if type(scatNrefrUnique) is np.ndarray:
            for i in range(len(scatNrefrUnique)):
                # If refractive index is smaller than 1e3, give different domain number, otherwise let the domain be 0 and use 
                # Dirichlet "wall" boundary condition to force wave to zero
                if scatNrefrUnique[i] < useDirichlet:
                    scatDom[scatNrefrAll == scatNrefrUnique[i]] = 4+i;
        scatDom = scatDom.astype(int)
        
        # If single radius given, convert to array
        if type(scatRad) is not np.ndarray:
            scatRad = scatRad*np.ones(numScat) 
  
        # Add scatterers
        for i in range(numScat):
            bcScat = 'scat'
            # Set boundary to wall such that it is treated as Dirichlet zero-boundary condition in Solve
            if scatDom[i] == 0:
                bcScat = 'wall'
            # If scatterer is embedded in other scatterer, change its rightdomain
            embedDom = 1 # Air domain
            if embedInd is not None and np.any(i == embedInd[0,:]):
                embedDom = scatDom[embedInd[1, i == embedInd[0,:]][0]]
            # Add circular scatterer
            if scatShape[i] == 'circ':
                geo.AddCircle((scatPos[0,i]+shiftLong[i], scatPos[1,i]+shiftTrans[i]), scatRad[i]+shiftRad[i],  leftdomain=scatDom[i], rightdomain=embedDom, bc=bcScat)
            # Add a circles which is used for the refinement of the mesh on the Gaussian potential
            elif scatShape[i] == 'gauss':
                geo.AddCircle((scatPos[0,i]+shiftLong[i], scatPos[1,i]+shiftTrans[i]), scatRad[i]+shiftRad[i],  leftdomain=scatDom[i], rightdomain=1, bc=bcScat)
            # Add square scatterers
            elif scatShape[i] == 'square':
                pts = np.array([[scatPos[0,i]-(scatRad[i]+shiftRad[i])+shiftLong[i], scatPos[0,i]+(scatRad[i]+shiftRad[i])+shiftLong[i], scatPos[0,i]+(scatRad[i]+shiftRad[i])+shiftLong[i], scatPos[0,i]-(scatRad[i]+shiftRad[i])+shiftLong[i]], [scatPos[1,i]-(scatRad[i]+shiftRad[i])+shiftTrans[i], scatPos[1,i]-(scatRad[i]+shiftRad[i])+shiftTrans[i], scatPos[1,i]+(scatRad[i]+shiftRad[i])+shiftTrans[i], scatPos[1,i]+(scatRad[i]+shiftRad[i])+shiftTrans[i]]])
                if shiftAng[i] != 0.0:
                    pts[0,:] -= scatPos[0,i]+shiftLong[i]
                    pts[1,:] -= scatPos[1,i]+shiftTrans[i]
                    rotMat = np.array([[np.cos(shiftAng[i]), -np.sin(shiftAng[i])],[np.sin(shiftAng[i]), np.cos(shiftAng[i])]])
                    pts = rotMat.dot(pts)
                    pts[0,:] += scatPos[0,i]+shiftLong[i]
                    pts[1,:] += scatPos[1,i]+shiftTrans[i]
                p1s,p2s,p3s,p4s = [ geo.AppendPoint(x,y,maxh,refine) for x,y,maxh,refine in [(pts[0,0],pts[1,0],maxhEdge,refVal), (pts[0,1],pts[1,1],maxhEdge,refVal), (pts[0,2],pts[1,2],maxhEdge,refVal), (pts[0,3],pts[1,3],maxhEdge,refVal)] ]
                geo.Append (["line", p1s, p2s], leftdomain=scatDom[i], rightdomain=embedDom, bc=bcScat)
                geo.Append (["line", p2s, p3s], leftdomain=scatDom[i], rightdomain=embedDom, bc=bcScat)
                geo.Append (["line", p3s, p4s], leftdomain=scatDom[i], rightdomain=embedDom, bc=bcScat)
                geo.Append (["line", p4s, p1s], leftdomain=scatDom[i], rightdomain=embedDom, bc=bcScat)
            # Add square scatterers with rounded edges
            elif scatShape[i] == 'squareRounded':
                dround = scatRad[i]/10.0
                xLE = scatPos[0,i]-(scatRad[i]+shiftRad[i])+shiftLong[i]
                xLR = scatPos[0,i]-(scatRad[i]+shiftRad[i])+shiftLong[i]+dround
                xRR = scatPos[0,i]+(scatRad[i]+shiftRad[i])+shiftLong[i]-dround
                xRE = scatPos[0,i]+(scatRad[i]+shiftRad[i])+shiftLong[i]
                yBE = scatPos[1,i]-(scatRad[i]+shiftRad[i])+shiftTrans[i]
                yBR = scatPos[1,i]-(scatRad[i]+shiftRad[i])+shiftTrans[i]+dround
                yTR = scatPos[1,i]+(scatRad[i]+shiftRad[i])+shiftTrans[i]-dround
                yTE = scatPos[1,i]+(scatRad[i]+shiftRad[i])+shiftTrans[i]
                pts = np.array([[xLE, xLR, xRR, xRE, xRE, xRE, xRE, xRR, xLR, xLE, xLE, xLE], [yBE, yBE, yBE, yBE, yBR, yTR, yTE, yTE, yTE, yTE, yTR, yBR]])
                if shiftAng[i] != 0.0:
                    pts[0,:] -= scatPos[0,i]+shiftLong[i]
                    pts[1,:] -= scatPos[1,i]+shiftTrans[i]
                    rotMat = np.array([[np.cos(shiftAng[i]), -np.sin(shiftAng[i])],[np.sin(shiftAng[i]), np.cos(shiftAng[i])]])
                    pts = rotMat.dot(pts)
                    pts[0,:] += scatPos[0,i]+shiftLong[i]
                    pts[1,:] += scatPos[1,i]+shiftTrans[i]
                p1s,p1sRr,p2sRl,p2s,p2sRr,p3sRr,p3s,p3sRl,p4sRr,p4s,p4sRl,p1sRl = [ geo.AppendPoint(x,y) for x,y in [(pts[0,0],pts[1,0]), (pts[0,1],pts[1,1]), (pts[0,2],pts[1,2]), (pts[0,3],pts[1,3]), (pts[0,4],pts[1,4]), (pts[0,5],pts[1,5]), (pts[0,6],pts[1,6]), (pts[0,7],pts[1,7]), (pts[0,8],pts[1,8]), (pts[0,9],pts[1,9]), (pts[0,10],pts[1,10]), (pts[0,11],pts[1,11])] ]
                geo.Append (["line", p1sRr, p2sRl], leftdomain=scatDom[i], rightdomain=embedDom, bc=bcScat)
                geo.Append (["line", p2sRr, p3sRr], leftdomain=scatDom[i], rightdomain=embedDom, bc=bcScat)
                geo.Append (["line", p3sRl, p4sRr], leftdomain=scatDom[i], rightdomain=embedDom, bc=bcScat)
                geo.Append (["line", p4sRl, p1sRl], leftdomain=scatDom[i], rightdomain=embedDom, bc=bcScat)
                geo.Append (["spline3", p1sRl, p1s, p1sRr], leftdomain=scatDom[i], rightdomain=embedDom, bc=bcScat)
                geo.Append (["spline3", p2sRl, p2s, p2sRr], leftdomain=scatDom[i], rightdomain=embedDom, bc=bcScat)
                geo.Append (["spline3", p3sRr, p3s, p3sRl], leftdomain=scatDom[i], rightdomain=embedDom, bc=bcScat)
                geo.Append (["spline3", p4sRr, p4s, p4sRl], leftdomain=scatDom[i], rightdomain=embedDom, bc=bcScat)
            # Add rectangular scatterers (their half width can be specified by rectWidth2, otherwise squares will be added)
            elif scatShape[i] == 'rect':                
                pts = np.array([[scatPos[0,i]-(scatRad[i]+shiftRad[i])+shiftLong[i], scatPos[0,i]+(scatRad[i]+shiftRad[i])+shiftLong[i], scatPos[0,i]+(scatRad[i]+shiftRad[i])+shiftLong[i], scatPos[0,i]-(scatRad[i]+shiftRad[i])+shiftLong[i]], [scatPos[1,i]-(rectWidth2[i]+shiftRad[i])+shiftTrans[i], scatPos[1,i]-(rectWidth2[i]+shiftRad[i])+shiftTrans[i], scatPos[1,i]+(rectWidth2[i]+shiftRad[i])+shiftTrans[i], scatPos[1,i]+(rectWidth2[i]+shiftRad[i])+shiftTrans[i]]])
                if shiftAng[i] != 0.0:
                    pts[0,:] -= scatPos[0,i]+shiftLong[i]
                    pts[1,:] -= scatPos[1,i]+shiftTrans[i]
                    rotMat = np.array([[np.cos(shiftAng[i]), -np.sin(shiftAng[i])],[np.sin(shiftAng[i]), np.cos(shiftAng[i])]])
                    pts = rotMat.dot(pts)
                    pts[0,:] += scatPos[0,i]+shiftLong[i]
                    pts[1,:] += scatPos[1,i]+shiftTrans[i]
                    
                p1s,p2s,p3s,p4s = [ geo.AppendPoint(x,y,maxh,refine) for x,y,maxh,refine in [(pts[0,0],pts[1,0],maxhEdge,refVal), (pts[0,1],pts[1,1],maxhEdge,refVal), (pts[0,2],pts[1,2],maxhEdge,refVal), (pts[0,3],pts[1,3],maxhEdge,refVal)] ]
                geo.Append (["line", p1s, p2s], leftdomain=scatDom[i], rightdomain=embedDom, bc=bcScat)
                geo.Append (["line", p2s, p3s], leftdomain=scatDom[i], rightdomain=embedDom, bc=bcScat)
                geo.Append (["line", p3s, p4s], leftdomain=scatDom[i], rightdomain=embedDom, bc=bcScat)
                geo.Append (["line", p4s, p1s], leftdomain=scatDom[i], rightdomain=embedDom, bc=bcScat)
            # Add chiral scatterers, i.e. two displaced half circles (no radius change implemented)
            elif scatShape[i] == 'chiral':
                offset = scatRad[i]/2.0
                pts = np.array([[scatPos[0,i]+shiftLong[i], scatPos[0,i]-scatRad[i]+shiftLong[i], scatPos[0,i]-scatRad[i]+shiftLong[i], scatPos[0,i]-scatRad[i]+shiftLong[i], scatPos[0,i]+shiftLong[i], scatPos[0,i]+shiftLong[i], scatPos[0,i]+scatRad[i]+shiftLong[i], scatPos[0,i]+scatRad[i]+shiftLong[i], scatPos[0,i]+scatRad[i]+shiftLong[i], scatPos[0,i]+shiftLong[i]], [scatPos[1,i]+scatRad[i]+offset+shiftTrans[i], scatPos[1,i]+scatRad[i]+offset+shiftTrans[i], scatPos[1,i]+offset+shiftTrans[i], scatPos[1,i]+offset-scatRad[i]+shiftTrans[i], scatPos[1,i]+offset-scatRad[i]+shiftTrans[i], scatPos[1,i]-offset-scatRad[i]+shiftTrans[i], scatPos[1,i]-offset-scatRad[i]+shiftTrans[i], scatPos[1,i]-offset+shiftTrans[i], scatPos[1,i]-offset+scatRad[i]+shiftTrans[i], scatPos[1,i]-offset+scatRad[i]+shiftTrans[i]]])
                if shiftAng[i] != 0.0:
                    pts[0,:] -= scatPos[0,i]+shiftLong[i]
                    pts[1,:] -= scatPos[1,i]+shiftTrans[i]
                    rotMat = np.array([[np.cos(shiftAng[i]), -np.sin(shiftAng[i])],[np.sin(shiftAng[i]), np.cos(shiftAng[i])]])
                    pts = rotMat.dot(pts)
                    pts[0,:] += scatPos[0,i]+shiftLong[i]
                    pts[1,:] += scatPos[1,i]+shiftTrans[i]
                pLC1,pLC2,pLC3,pLC4,pLC5,pRC1,pRC2,pRC3,pRC4,pRC5 = [ geo.AppendPoint(x,y) for x,y in [(pts[0,0],pts[1,0]), (pts[0,1],pts[1,1]), (pts[0,2],pts[1,2]), (pts[0,3],pts[1,3]), (pts[0,4],pts[1,4]), (pts[0,5],pts[1,5]), (pts[0,6],pts[1,6]), (pts[0,7],pts[1,7]), (pts[0,8],pts[1,8]), (pts[0,9],pts[1,9])] ]
                geo.Append (["spline3", pLC1, pLC2, pLC3], leftdomain=scatDom[i], rightdomain=embedDom, bc=bcScat)
                geo.Append (["spline3", pLC3, pLC4, pLC5], leftdomain=scatDom[i], rightdomain=embedDom, bc=bcScat)
                geo.Append (["line", pLC5, pRC1], leftdomain=scatDom[i], rightdomain=embedDom, bc=bcScat)
                geo.Append (["spline3", pRC1, pRC2, pRC3], leftdomain=scatDom[i], rightdomain=embedDom, bc=bcScat)
                geo.Append (["spline3", pRC3, pRC4, pRC5], leftdomain=scatDom[i], rightdomain=embedDom, bc=bcScat)
                geo.Append (["line", pRC5, pLC1], leftdomain=scatDom[i], rightdomain=embedDom, bc=bcScat)
            # Add polygon scatterer (no radius change or rotation implemented)
            elif scatShape[i] == 'poly':
                # Select polygon
                polyPts = polyPtsAll[polyCount]
                polyCodes = polyCodesAll[polyCount]
                polyDir = polyDirAll[polyCount]
                # Set domain numbers for counter-clockwise ('cccw') or clockwise ('cw') polygonal outline
                if polyDir == 'ccw':
                    leftdomain = scatDom[i]
                    rightdomain = embedDom
                elif polyDir == 'cw':
                    rightdomain = scatDom[i]
                    leftdomain = embedDom
                # Add straight line polygons
                if polyCodes is None:
                    # Check if polyVerts array contains the first vertex twice and remove it if this is the case
                    if np.array_equal(polyPts[:,0], polyPts[:,-1]):
                        polyPts = polyPts[:,:-1]
                    # Add vertex points to geometry
                    numVerts = np.shape(polyPts)[1]
                    polyVerts = [ geo.AppendPoint(scatPos[0,i]+polyPts[0,ip]+shiftLong[i], scatPos[1,i]+polyPts[1,ip]+shiftTrans[i], maxhEdge, refVal) for ip in range(numVerts) ]
                    # Connect all vertices with lines
                    for iv in range(numVerts-1):
                        geo.Append (["line", polyVerts[iv], polyVerts[iv+1]], leftdomain=leftdomain, rightdomain=rightdomain, bc=bcScat)
                    # Connect last vertex and first vertex
                    geo.Append (["line", polyVerts[-1], polyVerts[0]], leftdomain=leftdomain, rightdomain=rightdomain, bc=bcScat)
                # Add polygons which can also contain spline3 curves
                else:
                    if msgOutput is True:
                        print("rounded")
                    # Check if polyVerts array contains the first vertex twice and remove it if this is the case
                    if np.array_equal(polyPts[:,0], polyPts[:,-1]):
                        polyPts = polyPts[:,:-1]
                        polyCodes = polyCodes[:-1]
                    # Add vertex points to geometry
                    numVerts = np.shape(polyPts)[1]
                    polyVerts = [ geo.AppendPoint(scatPos[0,i]+polyPts[0,ip]+shiftLong[i], scatPos[1,i]+polyPts[1,ip]+shiftTrans[i], maxhEdge, refVal) for ip in range(numVerts) ]
                    iv = 0
                    while True:
                        # If we reach the last point and the first point has code 1 or 2, connect them with a straight line and break
                        if (iv == numVerts-1 and polyCodes[0] < 3):
                            geo.Append (["line", polyVerts[-1], polyVerts[0]], leftdomain=leftdomain, rightdomain=rightdomain, bc=bcScat)
                            break
                        # If we reach the point before the last and the next one has code 3, connect these two points and the first point with a spline3 curve and break
                        elif (iv == numVerts-2 and polyCodes[iv+1] == 3):
                            geo.Append (["spline3", polyVerts[iv], polyVerts[iv+1], polyVerts[0]], leftdomain=leftdomain, rightdomain=rightdomain, bc=bcScat)
                            break
                        else:
                            # Connect every 2 vertices with lines. polyCodes=1 is usually the starting point in any glyph, where 2 means connect by straight line.
                            if polyCodes[iv+1] < 3:
                                geo.Append (["line", polyVerts[iv], polyVerts[iv+1]], leftdomain=leftdomain, rightdomain=rightdomain, bc=bcScat)
                                iv += 1
                            # Connect every 3 points with bezier curve, i.e., a spline3
                            elif polyCodes[iv+1] == 3:
                                geo.Append (["spline3", polyVerts[iv], polyVerts[iv+1], polyVerts[iv+2]], leftdomain=leftdomain, rightdomain=rightdomain, bc=bcScat)
                                iv += 2 
                                if msgOutput is True:
                                    print("spline")
                # Increase polygon counter in order to add next polygon later on
                polyCount += 1    

                # numVert = np.shape(polyPts)[1]
                # polyVert = [ geo.AppendPoint(scatPos[0,i]+polyPts[0,ip]+shiftLong[i], scatPos[1,i]+polyPts[1,ip]+shiftTrans[i], maxhEdge, refVal) for ip in range(numVert) ]
                # # Connect all vertices with lines
                # for iv in range(numVert-1):
                #     geo.Append (["line", polyVert[iv], polyVert[iv+1]], leftdomain=scatDom[i], rightdomain=embedDom, bc=bcScat)
                # # Connect last vertex with first vertex
                # geo.Append (["line", polyVert[-1], polyVert[0]], leftdomain=scatDom[i], rightdomain=embedDom, bc=bcScat)

        # Assigning material names to domains
        geo.SetMaterial(1, "air")
        geo.SetMaterial(2, "pml")
        geo.SetMaterial(3, "pml")
        nmats = { "air" : uniformNr+1.0j*uniformNi, "pml" : uniformNr}
        for i in range(len(scatNrefrUnique[np.real(scatNrefrUnique < useDirichlet)])):
            geo.SetMaterial(4+i, "scat"+str(i))
            nmats["scat"+str(i)] = scatNrefrUnique[i] + (uniformNr-1.0)+1.0j*uniformNi

        # Set maximal height of triangles in domains of scatterers
        if meshScatRefine is True:
            for i in range(len(scatNrefrUnique)):
                if scatNrefrUnique[i] <= useDirichlet:
                    geo.SetDomainMaxH(4+i,maxhVac/np.real(nmats["scat"+str(i)]))

        # Create mesh
        with TaskManager():
            ngmesh = geo.GenerateMesh(maxh=maxhVac)
            mesh = Mesh(ngmesh)

        # Define coefficient function for refractive index using defined materials for predefined domains
        # or creating the Gaussian scattering potentials at the given positions where the placed circles 
        # above are then just used for local mesh refinement
        if scatShape[0] != 'gauss':
            ncoef = [nmats[mat] for mat in mesh.GetMaterials()]
            nrefr = CoefficientFunction(ncoef)
    
    # Empty waveguide        
    else:
        
        # Assigning material names to domains
        geo.SetMaterial(1, "air")
        geo.SetMaterial(2, "pml")
        geo.SetMaterial(3, "pml")
        
        # Create mesh
        with TaskManager():
            ngmesh = geo.GenerateMesh(maxh=maxhVac)
            mesh = Mesh(ngmesh)
        
        # Define coefficient function for refractive index using defined materials for predefined domains
        nmats = { "air" : uniformNr+1.0j*uniformNi, "pml" : uniformNr}
        ncoef = [nmats[mat] for mat in mesh.GetMaterials()]
        nrefr = CoefficientFunction(ncoef)

    # Set perfectly matched layers in Radial coordinates on domain 2
    pR = pml.Radial((0, 0), pmlRad, alpha=2j)
    mesh.SetPML(pR,2)

    # Refine mesh at the edge points for which refinement values are added
    mesh.RefineHP(refOrder, factor=refFactor)

    # Used curved elements at curved boundaries
    if meshCurve is True:
        mesh.Curve(3)
    # Manually set how the mesh should be curved
    elif meshCurve is not False and meshCurve is not None:
        mesh.Curve(meshCurve)

    # Output number of vertices in mesh
    if msgOutput is True:
        print('MESH number of vertices =', mesh.nv)
    
    # Reset inverted system matrix
    params['ainvGlobal'] = None
    params['mesh'] = mesh
    params['nrefr'] = nrefr
    params['nmats'] = nmats

def CreateState(params, coefs, deltaNorm=True, fluxNorm=False, dk=0.0):
    k = params['k']
    nIn = params['nIn']
    pmlRad = params['pmlRad']

    mode = lambda phi,n: exp(1.0j*n*phi)
        
    if deltaNorm:
        coefs = coefs*(-4.0j)/np.pi/pmlRad/sp.special.hankel1(np.arange(-nIn,nIn+1),(k+dk)*pmlRad)

    if fluxNorm:
        coefs *= np.sqrt(np.pi*pmlRad/2)
    
    r = sqrt(x*x + y*y)
    state = coefs[0] * mode(IfPos(y,acos(x/r),-acos(x/r)), -nIn)
    for i in range(-nIn+1,nIn+1):
        state += coefs[i+nIn] * mode(IfPos(y,acos(x/r),-acos(x/r)), i)
        
    return state

def Solve(params, coefs, dk=0, msgOutput=True, precond=None, deltaNorm=True, fluxNorm=False):
    
    # Convert required parameter dictionary entries to local variables
    feOrder = params['feOrder']
    k = params['k']
    mesh = params['mesh']
    nrefr = params['nrefr']
    nIn = params['nIn']
    
    # Retrieve k0
    if 'uniformNr' in params.keys():
        k0 = k/params['uniformNr']
    else:
        k0 = k

    ainvGlobal = params['ainvGlobal']

    # Check if a mesh exists (if not, Python will crash)
    if params['mesh'] is None:
        print('Mesh missing! Create mesh before using Solve!')
        return
    
    # For a single coefficient vector, create second array dimension for loop
    if np.ndim(coefs) == 1:
        coefs = coefs[:,np.newaxis]
        
    # OpenMP parallelize matrix assemblies, inner products, etc.
    with TaskManager():
    
        # Create complex finite element space with dirichlet boundary conditions at the waveguide walls
        fes = H1(mesh, complex=True, order=feOrder, dirichlet="wall")
        params['fes'] = fes
        u = fes.TrialFunction()
        v = fes.TestFunction()

        # Prints the degrees of freedom
        if msgOutput is True:
            print('FES degrees of freedom =', fes.ndof)
        
        # If inverted system matrix does not exist or the frequency is changed or there is mode dependent loss
        if ainvGlobal is None or np.abs(dk) > 0.0:

            # Create LHS of weak form of Helmholtz equation
            a = BilinearForm(fes, symmetric=True)
            a += SymbolicBFI(grad(u)*grad(v) )
            a += SymbolicBFI(-nrefr*nrefr*(k0+dk)*(k0+dk)*u*v)
            # Invert system matrix
            if precond is not None:
                print('Using \'' + precond + '\' preconditioner ...')
                c = Preconditioner(a, precond) # Register c to a BEFORE assembly, precond can be either 'local', 'direct', 'multigrid', 'h1amg' or 'bddc'
                a.Assemble()
                # Conjugate gradient solver
                ainv = CGSolver(a.mat, c.mat, maxsteps=1000)
            else:
                a.Assemble()
                ainv = a.mat.Inverse(freedofs=fes.FreeDofs(), inverse="sparsecholesky")
            # Save inverted system matrix if frequency shift is zero
            if (ainvGlobal is None and dk == 0.0) and precond is None:
                params['ainvGlobal'] = ainv
                if dk == 0.0 and msgOutput is True:
                    print('Stored inverted system matrix ...')
                elif msgOutput is True:
                    print('Temporarily stored inverted system matrix ...')  
        # Else use stored matrix from previous call of Solve
        else:
            # Only print that message in the case where there is no global loss since SolveModalLoss() calls Solve() many times
            if msgOutput is True:
                print('Using stored inverted system matrix ...')
            ainv = ainvGlobal
        
        # Get solutions of all coefficient vectors in mode basis by multiplying the source term with the inverse system matrix
        sols = []
        for i in range(np.shape(coefs)[1]):
            f = LinearForm(fes)
            sourceIn = CreateState(params, coefs[:,i], deltaNorm=deltaNorm, fluxNorm=fluxNorm, dk=dk)
            f += SymbolicLFI(sourceIn*v, BND, definedon=mesh.Boundaries("input"))
            f.Assemble()
            sol = GridFunction(fes)
            sol.vec.data = ainv * f.vec
            sols.append(sol)
        if len(sols) == 1:
            sols = sols[0]
        
    return sols

def CalcSMat(params, pphwl=20, fluxNorm=False, msgOutput=False, dk=0.):
    
    # Convert required parameter dictionary entries to local variables
    k = params['k']
    mesh = params['mesh']
    pmlRad = params['pmlRad']
    nIn = params['nIn']
    nOut = params['nOut']
    
    # Solve Helmholtz equation
    sols = Solve(params, np.eye(2*nIn+1), msgOutput=msgOutput, deltaNorm=True, fluxNorm=False, dk=dk)
    
    # Interpolate solutions to regular grid and calculate overlap integrals (gives slightly more unitary scattering matrices for high ppwhl)
    # Grid along the circular boundary
    npts = int(ceil(2*pmlRad*(k+dk)*pphwl))
    phis = np.linspace(0, 2*np.pi, num=npts)
    pCirc = np.zeros((npts, 2))
    pCirc[:,0] = pmlRad*np.cos(phis)
    pCirc[:,1] = pmlRad*np.sin(phis)

    modes = np.zeros((npts, 2*nOut+1), dtype=complex)

    solsOut = np.zeros((npts, 2*nIn+1), dtype=complex)

    # Interpolate the Hankel-modes at the boundary to the grid
    for n in range(-nOut, nOut+1):
        modes[:,n+nOut] = np.exp(1.0j*n*phis)/2/np.pi/sp.special.hankel1(n,(k+dk)*pmlRad)

    # Interpolate incoming and outgoing solutions to regular grid

    for m in range(2*nIn+1):
        for (i, phi) in enumerate(phis):
            solsOut[i,m] = sols[m](mesh(pmlRad*np.cos(phi), pmlRad*np.sin(phi)))

    S = np.zeros((2*nOut+1, 2*nIn+1), dtype=complex)

    for m in range(2*nOut+1):
        for n in range(2*nIn+1):
            S[m,n] = np.trapz(modes[:,m]*solsOut[:,n], x=phis)
            if m-nOut == -n+nIn:
                S[m,n] -= sp.special.hankel2(n-nOut,(k+dk)*pmlRad)/sp.special.hankel1(m-nOut,(k+dk)*pmlRad)

    return S

def CalcInputState(params, state, pphwl=20, returnPhis=False):
    nIn = params['nIn']
    k = params['k']
    pmlRad = params['pmlRad']
    
    npts = int(ceil(2*pmlRad*k*pphwl))
    phis = np.linspace(-np.pi,np.pi,num=npts)
    
    basis = lambda phi,n: np.exp(1.0j*n*phi)*sp.special.hankel2(n,k*pmlRad)/(np.sqrt(2*np.pi)*np.abs(sp.special.hankel1(n,k*pmlRad)))
    
    func = np.zeros(npts, dtype=complex)
    for i in range(0, 2*nIn+1):
        func += state[i] * basis(phis, i-nIn)
    
    if returnPhis is False:
        return func
    else:  
        return phis, func

def CalcPhiInOperator(params):
    k = params['k']
    pmlRad = params['pmlRad']
    nIn = params['nIn']
    nOut = params['nOut']
    
    phi = np.eye(2*nIn+1, dtype=complex) * np.pi
    
    c = np.exp(-1.0j*np.angle(sp.special.hankel1(np.arange(-nIn, nIn+1), k*pmlRad)))
    
    for m in range(-nOut,nOut+1):
        for n in range(-nIn,nIn+1):
            if m != n:
                phi[m+nOut,n+nIn] = 1.0j/(m-n)*np.conjugate(c[m+nOut])*c[n+nIn]
                
    phievals, phicoefs = np.linalg.eig(phi)
    
    isort = np.real(phievals).argsort()
    phievals = phievals[isort]
    phicoefs = phicoefs[:,isort]
    
    return phievals, phicoefs, phi

def CalcKxInOperator(params):
    k = params['k']
    nIn = params['nIn']
    pmlRad = params['pmlRad']
    
    hn = np.abs(sp.special.hankel1(np.arange(-nIn,nIn+1),k*pmlRad))
    kx = 1.j*np.diag(hn[:-1]/hn[1:] + hn[1:]/hn[:-1], k=-1) - 1.j*np.diag(hn[1:]/hn[:-1] + hn[:-1]/hn[1:], k=1)
    
    return k/4*kx

def CalcKyInOperator(params): # Check if correct
    k = params['k']
    nIn = params['nIn']
    pmlRad = params['pmlRad']
    
    hn = np.abs(sp.special.hankel1(np.arange(-nIn,nIn+1),k*pmlRad))
    ky = np.diag(hn[:-1]/hn[1:] + hn[1:]/hn[:-1], k=-1) + np.diag(hn[1:]/hn[:-1] + hn[:-1]/hn[1:], k=1)
    
    return k/4*ky

def CalcGWSOperator(params, S, dx, restrict=True, phicoefs=None, kIn=None, kOut=None, angle=np.pi/2, sort='real', deriv=True):
    
    nIn = params['nIn']

    if restrict is True:
        lowCut = int(nIn//(2*np.pi/angle) + 1)
        highCut = int(2*nIn-nIn//(2*np.pi/angle))
        if phicoefs is None:
            _, phicoefs, _ = CalcPhiInOperator(params)

    if deriv is True:

        if restrict is True:
            S_res = np.zeros((2*nIn+1,2*nIn+1,S.shape[2]), dtype=complex)

            for i in range(S.shape[2]):
                S_res[:,:,i] = phicoefs.T.conj() @ S[:,:,i] @ phicoefs
                S_res[:,lowCut:highCut,i] = 0.

            Q = -1.0j*S_res[:,:,2].conj().T@(S_res[:,:,1]-S_res[:,:,0])/(2.0*dx)

        else:
            Q = -1.0j*np.linalg.inv(S[:,:,2])@(S[:,:,1]-S[:,:,0])/(2.0*dx)

    else:

        if kIn is None:
            kIn = CalcKxInOperator(params)
        if kOut is None:
            kOut = CalcKxInOperator(params)    
        Q = kIn - S.conj().T @ kOut @ S

        if restrict is True:

            Q = phicoefs.conj().T @ Q @ phicoefs
            Q[:,lowCut:highCut] = 0.
            Q[lowCut:highCut,:] = 0.

    Qevals, Qcoefs = np.linalg.eig(Q)

    if sort == 'real':
        isort = np.real(Qevals).argsort()[::-1]
    elif sort == 'abs':
        isort = np.abs(Qevals).argsort()[::-1]
    Qevals = Qevals[isort]
    Qcoefs = Qcoefs[:,isort]

    if restrict is True:
        Qcoefs = phicoefs@Qcoefs

    return Qevals, Qcoefs, Q

def CalcForce(params, coefs, nums, scatPos, npts, scatRad=None, scatShape=None, polyPts=None, scatType='metal', d=1e-3, msgOutput=False):
    
    force = np.zeros((len(nums),2))
    mesh = params['mesh']

    sols = Solve(params, coefs[:,nums], msgOutput=msgOutput)
    
    for i in range(len(nums)):

        if scatType == 'metal':
            Egrad = Grad(sols[i])
        
        if scatShape == 'circle':
            # Set up grids at the boundary of the dielectric cylindrical scatterer
            phis = np.linspace(0,2.0*np.pi,npts)
            # Manually read out the state
            EgradVec = np.array([Egrad(mesh(scatPos[0]+(1+d)*scatRad*np.cos(phi), scatPos[1]+(1+d)*scatRad*np.sin(phi))) for phi in phis])
            f = np.abs(EgradVec[:,0])**2 + np.abs(EgradVec[:,1])**2
            # Manually integrate it
            force[i,0] = np.trapz(np.cos(phis)*f, x=phis, axis=0)
            force[i,1] = np.trapz(np.sin(phis)*f, x=phis, axis=0)

        if scatShape == 'poly':
            numLines = polyPts.shape[1]

            for n in range(numLines):
                nextPnt = (n + 1) % numLines
                x = np.linspace(scatPos[0]+polyPts[0,n], scatPos[0]+polyPts[0,nextPnt], npts)
                y = np.linspace(scatPos[1]+polyPts[1,n], scatPos[1]+polyPts[1,nextPnt], npts)

                xoff = polyPts[0,nextPnt] - polyPts[0,n]
                yoff = polyPts[1,nextPnt] - polyPts[1,n]
                
                normal = np.array([[0,1],[-1,0]])@np.array([xoff,yoff])

                if np.isclose(xoff,0.): # vertical line
                    spacing = np.abs(yoff) / (npts - 1)
                    ptsOut = np.vstack((x + np.sign(yoff)*d, y))
                    EgradVec = np.array([Egrad(mesh(ptsOut[0,i], ptsOut[1,i])) for i in range(len(ptsOut[0,:]))])
                    #EgradVec = Egrad(mesh(ptsOut[0,:], ptsOut[1,:]))
                    fx = np.abs(EgradVec[:,0])**2
                    force[i,0] += np.trapz(np.sign(normal[0])*fx, dx=spacing)

                elif np.isclose(yoff,0.): # horizontal line
                    spacing = np.abs(xoff) / (npts - 1)
                    ptsOut = np.vstack((x, y - np.sign(xoff)*d))
                    EgradVec = np.array([Egrad(mesh(ptsOut[0,i], ptsOut[1,i])) for i in range(len(ptsOut[0,:]))])
                    #EgradVec = Egrad(mesh(ptsOut[0,:], ptsOut[1,:]))
                    fy = np.abs(EgradVec[:,1])**2
                    force[i,1] += np.trapz(np.sign(normal[1])*fy, dx=spacing)

                else:
                    spacing = np.sqrt(xoff**2 + yoff**2) / (npts - 1)
                    ang = np.arctan(xoff / yoff)
                    dx = np.abs(d * np.cos(ang))
                    dy = np.abs(d * np.sin(ang))

                    if xoff > 0 and yoff > 0:
                        ptsOut = np.vstack((x + dx, y - dy))            
                    elif xoff > 0 and yoff < 0:
                        ptsOut = np.vstack((x - dx, y - dy))
                    elif xoff < 0 and yoff > 0:
                        ptsOut = np.vstack((x + dx, y + dy))
                    else:
                        ptsOut = np.vstack((x - dx, y + dy))

                    EgradVec = np.array([Egrad(mesh(ptsOut[0,i], ptsOut[1,i])) for i in range(len(ptsOut[0,:]))])
                    #EgradVec = Egrad(mesh(ptsOut[0,:], ptsOut[1,:]))
                    fx = np.abs(EgradVec[:,0])**2
                    fy = np.abs(EgradVec[:,1])**2

                    force[i,0] += np.trapz(np.sign(normal[0])*fx/np.cos(ang), dx=spacing)
                    force[i,1] += np.trapz(np.sign(normal[1])*fy/np.sin(ang), dx=spacing)
    
    return force

def PlotStates(params, coefs, nums, plot='abs2', method='imshow', interpolation='bilinear', cmap='jet', ctype='linear', cscale=None, clim=None, scatPos=np.array([]), 
    scatRad=0.0, scatShape='circ', scatLineWidth=None, pphwl=20, dpi=300, fileName=None, fileFormat='png', widthInches=20, rMinMax=None, phiMinMax=None, rectWidth2=None, 
    mCols=None, title=None, shiftAng=0.0, polyPts=None, polyCodes=None, msgOutput=False, returnData=False, colorBar=False, 
    colorBarLabel=True, colorBarTicks='MinMax', colorBarSize="1.5%", colorBarPad=0.25, colorBarLabelPad=15, fontSize=17):
    
    # Convert required parameter dictionary entries to local variables
    pmlRad = params['pmlRad']
    k = params['k']
    mesh = params['mesh']
    
    # Array with which you can shift the square/rectangular scatterers
    if np.size(scatPos) > 0:
        numScat = np.shape(scatPos)[1]
        if type(shiftAng) is not np.ndarray and shiftAng != 0.0:
            shiftAng = shiftAng*np.ones(numScat) 
        elif type(shiftAng) is not np.ndarray and shiftAng == 0.0:
            shiftAng = np.zeros(numScat) 
    else:
        numScat = 0

    # If scatShape is a single string, make a list out of it
    if isinstance(scatShape,list) is False:
        scatShape = [scatShape for i in range(numScat)]

    # Convert to array if it is a vector
    if type(coefs) is np.ndarray:
        if np.ndim(coefs) == 1:
            coefs = coefs[:,np.newaxis]
    # If it is a single given solution GridFunction, put it in a list
    elif type(coefs) is not list:
        coefs = [coefs] 
        nums = [0]
        
    # Set default plot range if no range is given
    if rMinMax is None:
        rMinMax = np.array([0,pmlRad])
    if phiMinMax is None:
        phiMinMax = np.array([0,2*np.pi])

    # Set up array for half rectangle widths
    if 'rect' in scatShape:
        if rectWidth2 is None:
            rectWidth2 = scatRad
        elif type(rectWidth2) is not np.ndarray:
            rectWidth2 = rectWidth2*np.ones(len(scatPos[0,:])) 

    # Set counter for polygons and put polyPts in a list in case there is only one polygon in order to be able to use the loop below
    numPoly = scatShape.count('poly')
    if isinstance(polyPts, list) is False:
        polyPtsAll = [polyPts for i in range(numPoly)]
    else:
        polyPtsAll = polyPts
    if isinstance(polyCodes, list) is False:
        polyCodesAll = [polyCodes for i in range(numPoly)]
    else:
        polyCodesAll = polyCodes

    # Define points for interpolation
    plotRadius = rMinMax[1]-rMinMax[0]
    plotAngle = phiMinMax[1]-phiMinMax[0]
    drRes = (2*np.pi/k)/2.0/float(pphwl)
    dxRes = (2*np.pi/k)/2.0/float(pphwl)
    dyRes = dxRes
    dphiRes = drRes
    Nr = int(np.ceil(plotRadius/drRes))
    Nphi = int(np.ceil(plotAngle/dphiRes))
    Nx = int(np.ceil(2*pmlRad/dxRes))
    rpts = np.array([rMinMax[0]+i*drRes for i in range(Nr+1)])
    phipts = np.array([phiMinMax[0]+i*dphiRes for i in range(Nphi+1)])
    xpts = xpts = np.linspace(-pmlRad,pmlRad,num=Nx)
    
    # Plot single image per state. If image has more than 2**16 pixels on one of the dimensions, Matplotlib cannot plot it as png, thus use pdf and convert it afterwards.
    #msgOutput = [False for n in range(len(nums))]
    #msgOutput[0] = True

    if returnData is True:
        stateGridReturn = np.zeros((Nx, Nx, len(nums)))
        
    for n in range(len(nums)):

        # Solve Helmholtz equation for coefficient vector
        if type(coefs) is np.ndarray:
            stateMesh = Solve(params, coefs[:,nums[n]], msgOutput=msgOutput)
        # Else, use GridFunction of solution list
        else:
            stateMesh = coefs[nums[n]]

        # Reads out values of the states on the regular grid. If point is not in mesh (outside the waveguide or for a hard scatterer)
        # NGSolve throws a warning in the terminal which slows the plotting down, but currently this cannot be resolved unless
        # one would check beforehand if point is inside the mesh, i.e., not in domain 0.
        stateGrid = np.zeros((Nx, Nx), dtype=complex)
        for ix in range(Nx):
            for iy in range(Nx):
                if rMinMax[0] <= np.sqrt(xpts[ix]**2 + xpts[iy]**2) <= rMinMax[1]:
                    if phiMinMax[0] < np.arctan2(xpts[iy], xpts[ix])%(2*np.pi) < phiMinMax[1]:
                        try:
                            stateGrid[iy,ix] = stateMesh(mesh(xpts[ix],xpts[iy]))
                        except:
                            pass
                    else:
                        stateGrid[iy,ix] = np.nan
                else:
                    stateGrid[iy,ix] = np.nan

        # Set up wave function array which will be plotted
        if plot == 'abs2':
            stateGrid = np.abs(stateGrid)**2
        elif plot == 'abs':
            stateGrid = np.abs(stateGrid)
        elif plot == 'real':
            stateGrid = np.real(stateGrid)
        elif plot == 'imag':
            stateGrid = np.imag(stateGrid)
        elif plot == 'phase':
            stateGrid = (np.angle(stateGrid) + 2.0*np.pi) % (2.0*np.pi)
        if cscale is not None:
            stateGrid = cscale*stateGrid
        if ctype == 'tanh':
            stateGrid = np.tanh(stateGrid)

        # Plot and show or save the resulting image
        plt.figure(figsize=(widthInches,widthInches))
        plt.gca().axis('off')
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.gca().axes.get_yaxis().set_visible(False)

        if method == 'imshow':
            hi = plt.imshow(stateGrid, origin='lower', interpolation=interpolation, cmap=cmap, extent=(-pmlRad,pmlRad,-pmlRad,pmlRad))
            if clim is not None:
                plt.clim(0,clim)
            # Draw boundary
            #circle = plt.Circle((0,0), radius=pmlRad, color='grey', ls='-', fill=False, linewidth=scatLineWidth)
            #plt.gca().add_artist(circle)
            # Draw scatterers
            polyCount = 0
            if numScat > 0:
                if type(scatRad) is not np.ndarray:
                    scatRad = scatRad*np.ones(numScat)
                for i in range(numScat):
                    # Define rotation matrix
                    if scatShape[i] == 'square' or scatShape[i] == 'rect':
                        rotMat = np.array([[np.cos(shiftAng[i]), -np.sin(shiftAng[i])],[np.sin(shiftAng[i]), np.cos(shiftAng[i])]])
                    #ix = (scatPos[0,i] + pmlRad)/dxRes
                    #iy = (scatPos[1,i] + pmlRad)/dxRes
                    if scatShape[i] == 'circ':
                        #irad = scatRad[i]/dxRes
                        circle = plt.Circle((scatPos[0,i],scatPos[1,i]), radius=scatRad[i], color='w', fill=False, linewidth=scatLineWidth)
                        plt.gca().add_artist(circle)
                    elif scatShape[i] == 'square':
                        #irad = scatRad[i]/dxRes
                        off = rotMat.dot(np.array([scatRad[i], scatRad[i]])) - np.array([scatRad[i], scatRad[i]]) # Offset due to rotation at the lower left corner point
                        square = plt.Rectangle((scatPos[0,i]-scatRad[i]-off[0],scatPos[1,i]-scatRad[i]-off[1]), 2*scatRad[i], 2*scatRad[i], angle=np.rad2deg(shiftAng[i]), color='w', fill=False, linewidth=scatLineWidth)
                        plt.gca().add_artist(square)
                    elif scatShape[i] == 'rect':
                        #irad = scatRad[i]/dxRes
                        #iwidth2 = rectWidth2[i]/dxRes
                        off = rotMat.dot(np.array([scatRad[i], rectWidth2[i]])) - np.array([scatRad[i], rectWidth2[i]]) # Offset due to rotation at the lower left corner point
                        rect = plt.Rectangle((scatPos[0,i]-scatRad[i]-off[0],scatPos[1,i]-rectWidth2[i]-off[1]), 2*scatRad[i], 2*rectWidth2, angle=np.rad2deg(shiftAng[i]), color='w', fill=False, linewidth=scatLineWidth)
                        plt.gca().add_artist(rect)
                    elif scatShape[i] == 'poly':
                        # Select polygon
                        polyPts = polyPtsAll[polyCount]
                        polyCodes = polyCodesAll[polyCount]
                        # Add polygon with straight line
                        if polyCodes is None:
                            ip = np.copy(polyPts)
                            ip[0,:] = scatPos[0,i] + polyPts[0,:]
                            ip[1,:] = scatPos[1,i] + polyPts[1,:]
                            poly = plt.Polygon(ip.T, ec="w", fill=False, linewidth=scatLineWidth)
                            plt.gca().add_patch(poly)
                        # Add polygon with curved segments defined by polyCodes
                        else:
                            polyPts = np.append(polyPts, polyPts[:,0][:,np.newaxis], axis=1)
                            polyCodes = np.append(polyCodes, Path.CLOSEPOLY)
                            ip = np.copy(polyPts)
                            params['pP'] = polyPts
                            ip[0,:] = ix + ip[0,:]/dxRes
                            ip[1,:] = iy + ip[1,:]/dyRes
                            path = Path(ip.T, polyCodes)
                            glyph = patches.PathPatch(path, ec="w", fill=False, linewidth=scatLineWidth)
                            plt.gca().add_patch(glyph)
                        polyCount += 1

        elif method == 'pcolormesh':
            [X,Y] = np.meshgrid(xpts, xpts)
            plt.pcolor(X, Y, statesGrid[:,:,n], cmap=cmap, shading='gouraud')
            plt.axes().set_aspect('equal')
        if colorBar is True:
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            ax = plt.gca()
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size=colorBarSize, pad=colorBarPad)
            cbar = plt.colorbar(hi, cax=cax, orientation="vertical")
            cbar.ax.tick_params(labelsize=fontSize)
            if colorBarLabel is True:
                if plot == 'abs2':
                    cbarLabel = r'$|\psi|^2$'
                elif plot == 'abs':
                    cbarLabel = r'$|\psi|$'
                elif plot == 'real':
                    cbarLabel = r'Re($\psi$)'
                elif plot == 'imag':
                    cbarLabel = r'Im($\psi$)'
                elif plot == 'phase':
                    cbarLabel = r'Arg($\psi$)'
                if colorBarTicks == 'MinMax':
                    cbar.set_label(cbarLabel, size=fontSize, labelpad=colorBarLabelPad-40)
                else:
                    cbar.set_label(cbarLabel, size=fontSize, labelpad=colorBarLabelPad)
                plt.clim([np.nanmin(stateGrid), np.nanmax(stateGrid)])
            if colorBarTicks is False:
                cbar.set_ticks([])
                cbar.set_ticklabels([])
            elif colorBarTicks == 'MinMax':
                cbar.set_ticks([np.nanmin(stateGrid), np.nanmax(stateGrid)])
                cbar.set_ticklabels(['Min', 'Max'])
        plt.tight_layout()

        # Show or save image
        if fileName is None:
            plt.show()
        else:
            if fileFormat == 'png' or fileFormat == 'pdf':
                plt.savefig(fileName+str(nums[n]+1)+'.'+fileFormat, transparent=True, bbox_inches='tight', dpi=dpi)
            else:
                plt.savefig(fileName+str(nums[n]+1)+'.'+fileFormat, format=fileFormat, bbox_inches='tight', dpi=dpi)
            plt.close()

        # Put read-out state into array for later return
        if returnData is True:
            stateGridReturn[:,:,n] = stateGrid

    if returnData is True:
        return stateGridReturn, xpts

def DrawRandPositions(params, radScat, fillFrac=None, numScat=0, radMax=None, shifts=None, 
                      scatShape='circ'):

    # Convert required parameter dictionary entries to local variables
    pmlRad = params['pmlRad']

    if (type(radScat) is not np.ndarray and 2*radScat > pmlRad) or 2*np.amax(radScat) > pmlRad:  
        print('Choose smaller scatterer!')
        return

    # Set up maximal distance to the center
    if radMax is None:
        radMax = pmlRad
    
    # If number of scatterers (for single radius) is not given, calculate it based on a given filling fraction
    if type(radScat) is not np.ndarray and numScat == 0 and fillFrac == None:
        posScats = np.array([[],[]])
        radScats = np.array([])
    else:
        if type(radScat) is not np.ndarray: 
            if fillFrac is not None:
                # Effective area occupied by scatterers
                Aeff =  radMax**2*np.pi
                # Calculate number of scatterers
                if scatShape == 'circ':
                    numScat = int(np.floor( fillFrac*Aeff/(np.pi*radScat**2) ))
            else:
                numScat = int(numScat)
            radScats = radScat*np.ones(numScat)
        # Randomly choose one of the radii
        else:
            # Fill the effective area with the same number of scatterers for each radius (which is equivalent with the average of randomly chosen radii)
            if fillFrac is not None:
                # Effective area occupied by scatterers
                Aeff =  radMax**2*np.pi
                # Calculate number of scatterers
                if scatShape == 'circ':
                    numScat = np.ones(len(radScat), dtype=int) * int(np.floor( fillFrac*Aeff/(np.pi*np.sum(radScat**2)) ))
                # Set up array for scatterer radii
                radScats = radScat[np.zeros(numScat[0]).astype(int)]
                for i in range(1,len(numScat)):
                    radScats = np.hstack((radScats, radScat[i*np.ones(numScat[i]).astype(int)]))
            # Randomly choose a radius numScat times
            elif type(numScat) is not np.ndarray:
                radScats = radScat[np.random.randint(np.size(radScat), size=numScat)]
            # Use the values in numScat for the given radii
            elif len(radScat) == len(numScat):
                # Sort such that the larger radii come first in the array and thus are added first (otherwise the algorithm gets
                # stuck since it cannot find a space to put the larger scatterers after already placing the smaller ones)
                isort = np.argsort(radScat)[::-1]
                radScat = radScat[isort]
                numScat = numScat[isort]
                numScat = numScat.astype(int)
                # Set up array for scatterer radii
                radScats = radScat[np.zeros(numScat[0]).astype(int)]
                for i in range(1,len(numScat)):
                    radScats = np.hstack((radScats, radScat[i*np.ones(numScat[i]).astype(int)]))

        numScat = np.sum(numScat)
        posScats = np.zeros((2,numScat))
        count = 0
        while count < numScat:
            
            # Randomly draw numbers in the cirular area bounded by radMax
            posTemp = np.array([radMax*(2*np.random.rand(1)-1), radMax*(2*np.random.rand(1)-1)])[:,0]
            
            distances = np.sqrt((posTemp[0] - posScats[0,:count])**2 + (posTemp[1] - posScats[1,:count])**2)
            # Check if scatterer is inside radMax
            if np.sqrt(posTemp[0]**2+posTemp[1]**2) > (radMax - radScats[count]):
                continue
            # Check if some of them overlap, i.e. if the distance between two center points is less than two radii (or sum of their different radii)
            elif count > 0 and np.any(distances < 2*np.amax(radScats)):
                if type(radScat) is not np.ndarray:
                    continue
                else:            
                    if np.any(distances[:count]-radScats[:count] < radScats[count]):
                        continue
                    else:
                        posScats[:,count] = posTemp
                        count += 1
            # Otherwise save them
            else:
                posScats[:,count] = posTemp
                count += 1

    return posScats, radScats

def PlotScatteringConfiguration(params, scatPos, scatRad, scatShape='circ', scatCol='r', showAxis=True, 
                                widthInches=20, rectWidth2=None, polyPts=None, fileName=None, dpi=300):

    pmlRad = params['pmlRad']

    numScat = np.shape(scatPos)[1]
    # Convert to list if it's just a single string
    if isinstance(scatShape,list) is False:
        scatShape = [scatShape for i in range(numScat)]
    # Define the array of half rectangle widths (or ellipse minor axis). If scatRad has a second dimension,
    # use it for rectWidth2 and use the first dimension for the stretch in x.
    if 'rect' in scatShape or 'ellipse' in scatShape:
        if np.ndim(scatRad) == 2:
            rectWidth2 = scatRad[1,:]
            scatRad = scatRad[0,:]
        else:
            if rectWidth2 is None:
                rectWidth2 = scatRad
            if type(rectWidth2) is not np.ndarray:
                rectWidth2 = rectWidth2*np.ones(numScat)
    # If single radius given, convert to array
    if type(scatRad) is not np.ndarray:
        scatRad = scatRad*np.ones(numScat) 

    # If single color string is given, convert it to a list
    if isinstance(scatCol,str) is True:
        scatCol = [scatCol for i in range(numScat)]
    # If a dictionary is given, convert it to list
    elif isinstance(scatCol,dict):
        scatCol = [scatCol[scatShape[i]] for i in range(numScat)]

    plt.figure(figsize=(widthInches,widthInches))
    # Plot the circular boundary
    circ = plt.Circle((0,0), radius=pmlRad, color='k', fill=False, ls='--', lw=2)
    plt.gca().add_artist(circ)
    # Then plot all different types of scatterers with corresponding colors
    for i in range(numScat):
        if scatShape[i] == 'circ':
            circ = plt.Circle((scatPos[0,i], scatPos[1,i]), radius=scatRad[i], color=scatCol[i])
            plt.gca().add_artist(circ)
        elif scatShape[i] == 'square':
            square = plt.Rectangle((scatPos[0,i]-scatRad[i], scatPos[1,i]-scatRad[i]), 2*scatRad[i], 2*scatRad[i], color=scatCol[i])
            plt.gca().add_artist(square)
        elif scatShape[i] == 'rect':
            rect = plt.Rectangle((scatPos[0,i]-scatRad[i], scatPos[1,i]-rectWidth2[i]), 2*scatRad[i], 2*rectWidth2[i], color=scatCol[i])
            plt.gca().add_artist(rect)
        elif scatShape[i] == 'poly':
            plotPts = np.copy(polyPts)
            plotPts[0,:] += scatPos[0,i]
            plotPts[1,:] += scatPos[1,i]
            poly = plt.Polygon(plotPts.T, color=scatCol[i])
            plt.gca().add_patch(poly) 
    plt.xlim([-1.01*pmlRad,1.01*pmlRad])
    plt.ylim([-1.01*pmlRad,1.01*pmlRad])
    plt.tight_layout()
    plt.gca().set_aspect('equal')
    if showAxis is False:
        plt.axis('off')

    # Save image or show plot
    if fileName is not None:
        plt.savefig(fileName+'.png', bbox_inches='tight', dpi=dpi)
    else:
        plt.show()

def CreateMesh(params, scatPos=np.array([[],[]]), scatRad=np.array([]), scatNr=1.0, scatNrLast=None, scatNi=0.0, scatNiLast=None, 
               uniformNr=1.0, uniformNi=0.0, sqphwl=None, ROI=None, ROICenter=None, pixNr=None, pmlLength=None, msgOutput=True, meshScatRefine=True, 
               meshEdgeRefine=True, meshCurve=True):

    """
    Create a scattering geometry consisting of a circular scattering region bounded by a perfectly matched layer (PML) region to absorb all outgoing waves
    """

    # Convert required parameter dictionary entries to local variables
    nphwl = params['nphwl']
    k = params['k']
    feOrder = params['feOrder']

    numScat = len(scatPos[0,:])

    if type(scatNr) is not np.ndarray and scatNr != 1.0:
        scatNr = scatNr*np.ones(numScat) 
    elif type(scatNr) is not np.ndarray and scatNr == 1.0:
        scatNr = np.ones(numScat) 
    if scatNrLast != None:
        scatNr[-1] = scatNrLast
            
    if type(scatNi) is not np.ndarray and scatNi != 0.0:
        scatNi = scatNi*np.ones(numScat) 
    elif type(scatNi) is not np.ndarray and scatNi == 0.0:
        scatNi = np.zeros(numScat) 
    if scatNiLast != None:
        scatNi[-1] = scatNiLast

    # Values used for refinement of edges in the empty geometry or of polygonal scatterers
    maxhVac = (2*np.pi/k)/2.0/nphwl
    if meshEdgeRefine is True:
        maxhEdge = maxhVac/5.0
    else:
        maxhEdge = maxhVac
    refVal = 0
    refOrder = 3
    refFactor = 0.5

    # If not given, choose length of PML as nlambda times the longest wavelength in propagation direction which gets dampened the least
    if pmlLength is None:
        nlambda = 1.5
        lambdaMax = 2.0*np.pi/k
        pmlLength = nlambda*lambdaMax
        
    if 'pmlRad' in params.keys():
        pmlRad = params['pmlRad']
    else:
        pmlRad = 0.1
        params['pmlRad'] = pmlRad

    # Define edge points for waveguide boundary and input and output PML
    geo = SplineGeometry()
    
    geo.AddCircle((0, 0), pmlRad,  leftdomain=1, rightdomain=2, bc="input")
    geo.AddCircle((0, 0), pmlRad+pmlLength,  leftdomain=2, rightdomain=0, bc="wall")
    
    # Add points for pixel structure
    if sqphwl is not None:
        dyRes = (2*np.pi/np.real(k))/2.0/float(sqphwl)
        dxRes = dyRes
        Ny = int(np.ceil(ROI[1]/dyRes))
        Nx = int(np.ceil(ROI[0]/dxRes))
        dxRes = ROI[0]/Nx
        dyRes = ROI[1]/Ny
        params['Ny'] = Ny
        params['Nx'] = Nx
    
        scatDom = np.arange(4,4+Nx*Ny).reshape(Nx,Ny)
    
        for i in range(Nx):
            for j in range(Ny):
                if j == 0:
                    leftdomain1 = scatDom[i,j]
                    rightdomain1 = 1
                else:
                    leftdomain1 = scatDom[i,j]
                    rightdomain1 = scatDom[i,j] - 1
                if i == 0:
                    leftdomain2 = 1
                    rightdomain2 = scatDom[i,j]
                else:
                    leftdomain2 = scatDom[i,j] - Ny
                    rightdomain2 = scatDom[i,j]
    
                xposLR = dxRes*np.array([i,i+1]) + ROICenter[0] - Nx/2*dxRes
                yposLR = dyRes*np.array([j,j]) + ROICenter[1] - Ny/2*dyRes
    
                geo.Append (["line", geo.AppendPoint(xposLR[0], yposLR[0]), 
                             geo.AppendPoint(xposLR[1], yposLR[1])], leftdomain=leftdomain1, 
                            rightdomain=rightdomain1, bc='scat')
    
                xposUD = dxRes*np.array([i,i]) + ROICenter[0] - Nx/2*dxRes
                yposUD = dyRes*np.array([j,j+1]) + ROICenter[1] - Ny/2*dyRes
    
                geo.Append (["line", geo.AppendPoint(xposUD[0], yposUD[0]), 
                             geo.AppendPoint(xposUD[1], yposUD[1])], leftdomain=leftdomain2, 
                            rightdomain=rightdomain2, bc='scat')
    
        for i in range(Nx):
                xposU = dxRes*np.array([i,i+1]) + ROICenter[0] - Nx/2*dxRes
                yposU = dyRes*np.array([Ny,Ny]) + ROICenter[1] - Ny/2*dyRes
                geo.Append (["line", geo.AppendPoint(xposU[0], yposU[0]), 
                             geo.AppendPoint(xposU[1], yposU[1])], leftdomain=1, 
                            rightdomain=scatDom[i,-1], bc='scat')
    
        for i in range(Ny):
                xposR = dxRes*np.array([Nx,Nx]) + ROICenter[0] - Nx/2*dxRes
                yposR = dyRes*np.array([i,i+1]) + ROICenter[1] - Ny/2*dyRes
                geo.Append (["line", geo.AppendPoint(xposR[0], yposR[0]), 
                             geo.AppendPoint(xposR[1], yposR[1])], leftdomain=scatDom[-1,i], 
                            rightdomain=1, bc='scat')
        
    # Add randomly placed scatterers
    if numScat > 0:
        
        # Create domain numbers for different scatterers using their refractive index (including the shifts). Since np.unique
        # returns a sorted array, the metallic scatterers with index > 1e3 are the last and will be given domain 0
        useDirichlet = 1e3
        scatNrefrAll = scatNr + 1.0j*scatNi
        scatNrefrUnique = np.unique(scatNrefrAll)
        scatDom = np.zeros(numScat)
        if type(scatNrefrUnique) is np.ndarray:
            for i in range(len(scatNrefrUnique)):
                # If refractive index is smaller than 1e3, give different domain number, otherwise let the domain be 0 and use 
                # Dirichlet "wall" boundary condition to force wave to zero
                if scatNrefrUnique[i] < useDirichlet:
                    scatDom[scatNrefrAll == scatNrefrUnique[i]] = 4+Nx*Ny+i;
        scatDom = scatDom.astype(int)
        
        # If single radius given, convert to array
        if type(scatRad) is not np.ndarray:
            scatRad = scatRad*np.ones(numScat) 
  
        # Add scatterers
        for i in range(numScat):
            bcScat = 'scat'
            # Set boundary to wall such that it is treated as Dirichlet zero-boundary condition in Solve
            if scatDom[i] == 0:
                bcScat = 'wall'
            geo.AddCircle((scatPos[0,i], scatPos[1,i]), scatRad[i],  leftdomain=scatDom[i], rightdomain=1, bc=bcScat)

        # Assigning material names to domains
        geo.SetMaterial(1, "air")
        geo.SetMaterial(2, "pml")
        geo.SetMaterial(3, "pml")
        
        nmats = { "air" : uniformNr+1.0j*uniformNi, "pml" : uniformNr}

        # Set refractive index of pixels
        for i in range(Nx*Ny):
            geo.SetMaterial(4+i, "pix"+str(i))
            nmats["pix"+str(i)] = pixNr[i]
            
        for i in range(len(scatNrefrUnique[np.real(scatNrefrUnique < useDirichlet)])):
            geo.SetMaterial(4+Nx*Ny+i, "scat"+str(i))
            nmats["scat"+str(i)] = scatNrefrUnique[i] + (uniformNr-1.0)+1.0j*uniformNi

        # Set maximal height of triangles in domains of scatterers
        if meshScatRefine is True:
            for i in range(len(scatNrefrUnique)):
                if scatNrefrUnique[i] <= useDirichlet:
                    geo.SetDomainMaxH(4+Nx*Ny+i,maxhVac/np.real(nmats["scat"+str(i)]))

        # Create mesh
        with TaskManager():
            ngmesh = geo.GenerateMesh(maxh=maxhVac)
            mesh = Mesh(ngmesh)
        
        # Define coefficient function for refractive index using defined materials for predefined domains
        ncoef = [nmats[mat] for mat in mesh.GetMaterials()]
        nrefr = CoefficientFunction(ncoef)
    
    # Empty waveguide        
    else:
        
        # Assigning material names to domains
        geo.SetMaterial(1, "air")
        geo.SetMaterial(2, "pml")
        geo.SetMaterial(3, "pml")

        # Define coefficient function for refractive index using defined materials for predefined domains
        nmats = { "air" : uniformNr+1.0j*uniformNi, "pml" : uniformNr}

        if len(pixNr) != Nx*Ny:
            pixNr = np.hstack((pixNr, np.ones(Nx*Ny - len(pixNr))))
            
        # Set refractive index of pixels
        for i in range(Nx*Ny):
            geo.SetMaterial(4+i, "pix"+str(i))
            nmats["pix"+str(i)] = pixNr[i]

        # Create mesh
        with TaskManager():
            ngmesh = geo.GenerateMesh(maxh=maxhVac)
            mesh = Mesh(ngmesh)
         
        ncoef = [nmats[mat] for mat in mesh.GetMaterials()]
        nrefr = CoefficientFunction(ncoef)
        
    # Set perfectly matched layers in Radial coordinates on domain 2
    pR = pml.Radial((0, 0), pmlRad, alpha=2j)
    mesh.SetPML(pR,2)

    # Refine mesh at the edge points for which refinement values are added
    mesh.RefineHP(refOrder, factor=refFactor)

    # Used curved elements at curved boundaries
    if meshCurve is True:
        mesh.Curve(3)
    # Manually set how the mesh should be curved
    elif meshCurve is not False and meshCurve is not None:
        mesh.Curve(meshCurve)

    # Output number of vertices in mesh
    if msgOutput is True:
        print('MESH number of vertices =', mesh.nv)
    
    # Reset inverted system matrix
    params['ainvGlobal'] = None
    params['mesh'] = mesh
    params['nrefr'] = nrefr
    params['nmats'] = nmats

def updateNrefr(params, pixNr):
    
    nmats = params['nmats']
    mesh = params['mesh']
    
    # Set the material's new refractive index and create new CoefficientFunction
    for i in range(len(pixNr)):
        nmats['pix'+str(i)] = pixNr[i]
    
    ncoef = [nmats[mat] for mat in mesh.GetMaterials()]
    nrefr = CoefficientFunction(ncoef)
    
    # Reset the stored inverted system matrix
    params['ainvGlobal'] = None
    params['nrefr'] = nrefr
    params['nmats'] = nmats

def CalcOutputState(params, state, pphwl=20, returnPhis=False):
    nIn = params['nIn']
    k = params['k']
    pmlRad = params['pmlRad']
    
    npts = int(ceil(2*pmlRad*k*pphwl))
    phis = np.linspace(-np.pi,np.pi,num=npts)
    
    basis = lambda phi,n: np.exp(-1.0j*n*phi)*sp.special.hankel1(n,k*pmlRad)/(np.sqrt(2*np.pi)*np.abs(sp.special.hankel1(n,k*pmlRad)))
    
    func = np.zeros(npts, dtype=complex)
    for i in range(0, 2*nIn+1):
        func += state[i] * basis(phis, i-nIn)
    
    if returnPhis is False:
        return func
    else:  
        return phis, func

def CalcPhiOutOperator(params):
    k = params['k']
    pmlRad = params['pmlRad']
    nIn = params['nIn']
    nOut = params['nOut']
    
    phi = np.eye(2*nIn+1, dtype=complex) * np.pi
    
    c = np.exp(1.0j*np.angle(sp.special.hankel1(np.arange(-nIn, nIn+1), k*pmlRad)))
    
    for m in range(-nOut,nOut+1):
        for n in range(-nIn,nIn+1):
            if m != n:
                phi[m+nOut,n+nIn] = -1.0j/(m-n)*np.conjugate(c[m+nOut])*c[n+nIn]
                
    phievals, phicoefs = np.linalg.eig(phi)
    
    isort = np.real(phievals).argsort()
    phievals = phievals[isort]
    phicoefs = phicoefs[:,isort]
    
    return phievals, phicoefs, phi

def calcQsROI(params, ROI, ROICenter, sqphwl=None, timer=False):
    nIn = params['nIn']
    mesh = params['mesh']
    k = params['k']
    
    if sqphwl is not None:
        dyRes = (2*np.pi/np.real(k))/2.0/float(sqphwl)
        dxRes = dyRes
        Ny = int(np.ceil(ROI[1]/dyRes))
        Nx = int(np.ceil(ROI[0]/dxRes))

    xs = np.linspace(ROICenter[0]-ROI[0]/2+ROI[0]/Nx/2, ROICenter[0]+ROI[0]/2-ROI[0]/Nx/2, num=Nx)
    ys = np.linspace(ROICenter[1]-ROI[1]/2+ROI[1]/Ny/2, ROICenter[1]+ROI[1]/2-ROI[1]/Ny/2, num=Ny)

    XS, YS = np.meshgrid(xs,ys)
    XS_flat = XS.T.flatten()
    YS_flat = YS.T.flatten()
    
    Qn = np.zeros((Nx*Ny,2*nIn+1,2*nIn+1), dtype=complex)
    
    if timer == True:
        st = timeit.default_timer()
    sols = Solve(params, np.eye(2*nIn+1), msgOutput=False)
        
    stateGrid = np.zeros((2*nIn+1, Nx*Ny), dtype=complex)
    for i in range(2*nIn+1):
        for j in range(Nx*Ny):
            stateGrid[i,j] = sols[i](mesh(XS_flat[j],YS_flat[j]))
        
    if timer == True: 
        print('EM-Simulations && Interpolation: %fs' %(timeit.default_timer() - st))
        st = timeit.default_timer()
        
    for i in range(2*nIn+1):
        for j in range(i,2*nIn+1):
            if i == j:
                Qn[:,i,j] = np.abs(stateGrid[i,:])**2
            else:
                I = np.abs(stateGrid[i,:] + stateGrid[j,:])**2 + 0.j
                I -= np.abs(stateGrid[i,:] - stateGrid[j,:])**2
                I += 1.0j*np.abs(stateGrid[i,:] - 1.0j*stateGrid[j,:])**2
                I -= 1.0j*np.abs(stateGrid[i,:] + 1.0j*stateGrid[j,:])**2
                
                Qn[:,i,j] = I/4
                Qn[:,j,i] = np.conjugate(I)/4
    
    if timer == True:
        print('Calc Qs: %fs' %(timeit.default_timer() - st))
        
    #return k**2*Qn/2
    return Qn