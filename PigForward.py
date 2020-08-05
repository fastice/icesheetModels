#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
from utilities import myerror
import modelfunc as mf
import firedrake
import icepack
from icepack.constants import ice_density as rhoI, water_density as rhoW
from firedrake import max_value, min_value
from datetime import datetime
import rasterio
import icepack.inverse
from firedrake import grad
from icepack.constants import gravity as g
from icepack.constants import weertman_sliding_law as m
from icepack.constants import glen_flow_law as n
import matplotlib.pyplot as plt
import icepack.plot
import icepack.models
from firedrake import PETSc
import numpy as np
import yaml
import os

floatingG, groundedG, mesh = None, None, None


def parsePigForwardArgs():
    ''' Handle command line args'''
    defaults = {'geometry': 'PigGeometry.yaml',
                'degree': 1,
                'plotResult': False,
                'params': None,
                'inversionResult': None,
                'nYears': 10.,
                'GLThresh': 100,
                'SMB': '/home/ian/ModelRuns/Thwaites/BrookesMap/'
                    'OLS_Trend_plus_Resid_9b9.tif',
                'deltaT': 0.05,
                'meltParams': 'meltParams.yaml'
                }
    parser = argparse.ArgumentParser(
        description='\n\n\033[1mRun a forward simulation initialized by an '
        'inversion \033[0m\n\n')
    parser.add_argument('--geometry', type=str, default=None,
                        help=f'Yaml file with geometry file info '
                        f'[{defaults["geometry"]}] ')
    parser.add_argument('--degree', type=int, default=None,
                        choices=[1, 2],
                        help=f'Degree for mesh ')
    parser.add_argument('--nYears', type=float, default=None,
                        help=f'Simulation length (yrs) [{defaults["nYears"]}]')
    parser.add_argument('--deltaT', type=float, default=None,
                        help=f'Time step (yrs) [{defaults["deltaT"]}]')
    parser.add_argument('--plotResult', action='store_true',
                        default=None,
                        help=f'Display results [{defaults["plotResult"]}]')
    parser.add_argument('--params', type=str, default=None,
                        help=f'Input parameter file (.yaml)'
                        f'[{defaults["params"]}]')
    parser.add_argument('inversionResult', type=str, nargs=1,
                        help=f'Base name(.yaml/.h5) with inversion result')
    parser.add_argument('forwardResult', type=str, nargs=1,
                        help=f'Base name forward output')
    #
    forwardParams, inversionParams = parseForwardParams(parser, defaults)
    PETSc.Sys.Print('\n\n**** FORWARD MODEL PARAMS ****')
    for key in forwardParams:
        PETSc.Sys.Print(f'{key}: {forwardParams[key]}')
    PETSc.Sys.Print('**** END MODEL PARAMS ****\n')
    #
    return forwardParams, inversionParams


def parseForwardParams(parser, defaults):
    """
    Parse model params with the following precedence:
    1) Set at command line,
    2) Set in a parameter file,
    3) Default value.
    Merge in parameters that are taken from inversion result
    """
    #
    args = parser.parse_args()
    # Read file
    forwardParams = mf.readModelParams(args.params, key='forwardParams')  
    for arg in vars(args):
        # If value input through command line, override existing.
        argVal = getattr(args, arg)
        if argVal is not None:
            forwardParams[arg] = argVal
    for key in defaults:
        if key not in forwardParams:
            forwardParams[key] = defaults[key]
    # get rid of lists for main args
    forwardParams['inversionResult'] = forwardParams['inversionResult'][0]
    forwardParams['forwardResult'] = forwardParams['forwardResult'][0]
    # read inversonParams
    inversionYaml = f'{forwardParams["inversionResult"]}.yaml'
    inversionParams = mf.readModelParams(inversionYaml, key='inversionParams')
    #
    # Grap inversion params for forward sim
    for key in ['friction', 'degree', 'mesh', 'uThresh']:
        try:
            forwardParams[key] = inversionParams[key]
        except Exception:
            myerror(f'parameter- {key} - missing from inversion result')
    return forwardParams, inversionParams


def getInversionData(forwardParams, Q, V):
    """ Read and return inversion data"""
    #
    inversionData = ['betaInv', 'AInv', 'sInv', 'hInv', 'zbInv', 'floatingInv',
                     'groundedInv']
    myVars = mf.getCheckPointVars(forwardParams['inversionResult'],
                                  inversionData, Q)
    myList = [myVars[myVar] for myVar in inversionData] 
    uObsInv = mf.getCheckPointVars(forwardParams['inversionResult'],
                                   ['uObsInv'], V)['uObsInv']
    # betaInv,Ainv,  sInv, hInv, zbInv, floatingInv,groundedInv,uObsInv
    myList.append(uObsInv)
    return myList[:]


def viscosityNoTheta(u, h, grounded, A):
    return icepack.models.viscosity.viscosity_depth_averaged(u, h, A)


def velocityError(uO, uI, area, message=''):
    """
    Compute and print velocity error
    """
    deltaV = uO - uI
    vError = firedrake.inner(deltaV, deltaV)
    vErrorAvg = np.sqrt(firedrake.assemble(vError * firedrake.dx) / area)
    PETSc.Sys.Print(f'{message} v error {vErrorAvg:10.2f} (m/yr)\n')
    return vErrorAvg.item()

def main():
    # declare globals - fix later
    global mesh, floatingG, groundedG 
    forwardParams, inversionParameters = parsePigForwardArgs()
    #
    # Read mesh and setup function spaces
    mesh, Q, V, meshOpts = mf.setupMesh(forwardParams['mesh'],
                                        degree=forwardParams['degree'])
                            
    opts = {}
    opts['dirichlet_ids'] = meshOpts['dirichlet_ids'] # Opts from mesh
    #
    beta, A, s0, h0, zb, floating0, grounded0, uObs = \
        getInversionData(forwardParams, Q, V)
    SMB = mf.getModelVarFromTiff(forwardParams['SMB'], Q)
    SMB = icepack.interpolate(firedrake.max_value(firedrake.min_value(SMB,2),0), Q)
    meltParams = mf.inputMeltParams(forwardParams['meltParams'])
    #
    area = firedrake.assemble(firedrake.Constant(1) * firedrake.dx(mesh))
    areaG0 = firedrake.assemble(grounded0 * firedrake.dx(mesh))
    areaF0 = area - areaG0
    #
    # Setup ice stream model
    try: 
        frictionLaw = {'weertman': mf.weertmanFriction,
                       'schoof': mf.schoofFriction}[forwardParams['friction']]
    except Exception:
        myerror(f'Invalid friction law: {forwardParams["friction"]}')
    #
    forwardModel = icepack.models.IceStream(friction=frictionLaw,
                                            viscosity=viscosityNoTheta)
    forwardSolver = icepack.solvers.FlowSolver(forwardModel, **opts)
    # initial solve
    u0 = forwardSolver.diagnostic_solve(u=uObs, h=h0, s=s0, A=A, beta=beta,
                                        grounded=grounded0, floating=floating0,
                                        uThresh=forwardParams['uThresh'])
    # copy original state
    h, s = h0.copy(deepcopy=True), s0.copy(deepcopy=True)
    u = u0.copy(deepcopy=True)
    zF = mf.flotationHeight(zb, Q)
    grounded = grounded0.copy(deepcopy=True)
    floating = floating0.copy(deepcopy=True)
    #
    figU, axesU = icepack.plot.subplots(1, 3)
    #
    beginTime = datetime.now()
    hLast = h.copy(deepcopy=True)
    for t in np.arange(0, forwardParams['nYears'], forwardParams['deltaT']):
        #
        melt = mf.piecewiseWithDepth(h, floating, meltParams)
        a = SMB + icepack.interpolate(melt, Q)
        h = forwardSolver.prognostic_solve(forwardParams['deltaT'], h=h, u=u,
                                           a=a, h_inflow=h0)

        # NEED TO MODIFY THIS FUNCTION TO UPDATE GROUNDED AND FLOATING                                  
        s = icepack.compute_surface(h=h, b=zb)
        # NEED TO APPLY BETA SCALE
        betaScale = mf.reduceNearGLBeta(s, s0, zF, grounded, Q, forwardParams['GLThresh'])
        # NEED TO ADD MELT FUNCTION
        #
        u = forwardSolver.diagnostic_solve(u=u, h=h, s=s, A=A, beta=beta,
                                           grounded=grounded, floating=floating,
                                           uThresh=forwardParams['uThresh'])
        areaG = firedrake.assemble(grounded * firedrake.dx(mesh))
        areaF = area - areaG
        #
        pStep = 5
        if t % pStep == 0 and t > 0.5:
            for j in range(0,3):
                axesU[j].clear()
            
            dS = icepack.interpolate((h - hLast), Q)
            levels = np.linspace(-4, 4, 401)
            dhCont = icepack.plot.tricontourf(dS, levels=levels,
                                      extend='max', axes=axesU[0], cmap=plt.get_cmap('bwr'))
            
            sCont = icepack.plot.tricontourf(s, levels=np.linspace(0, 2000, 201),
                                      extend='max', axes=axesU[1])
        
            speed = firedrake.sqrt(firedrake.inner(u, u))
            speed = firedrake.interpolate(speed, Q)
            cSpeed = icepack.plot.tricontourf(speed, levels=np.linspace(0,4000, 400),
                                              extend='max', axes=axesU[2])                        
            if t < pStep+1:  # Just first time through                        
                figU.colorbar(dhCont, ax=axesU[0])
                figU.colorbar(sCont, ax=axesU[1])
                figU.colorbar(cSpeed, ax=axesU[2]) 

            plt.draw()
            plt.pause(0.1)
        if t % 1 == 0 and t > 0.5:
            print(f'year {t} runtime {datetime.now() - beginTime}')
            volumeChangeG = firedrake.assemble(grounded * (h - hLast) * firedrake.dx(mesh))
            volumeChangeF = firedrake.assemble(floating * (h - hLast) * firedrake.dx(mesh))
            print(f'Average change grounded {volumeChangeG/areaG:0.4f} in km/3 {volumeChangeG/1e9:0.2f}')
            print(f'Average change floating {volumeChangeF/areaG:0.4f} in km/3 {volumeChangeF/1e9:0.2f}')
            velocityError(u, u0, area)
            hLast = h.copy(deepcopy=True)
    
    speed = firedrake.sqrt(firedrake.inner(u, u))
    speed = firedrake.interpolate(speed, Q)
   
    
    plt.show()

main()
  
