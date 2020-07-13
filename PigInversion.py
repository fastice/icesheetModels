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

sigmaX, sigmaY, uObs, mesh = None, None, None, None
floatingG, groundedG = None, None


def setupPigInversionArgs():
    ''' Handle command line args'''
    defaults = {'geometry': 'PigGeometry.yaml',
                'velocity':
                '/home/ian/ModelRuns/PIG2018/Data/velocity/pseudo2000/vQ2000',
                'mesh': 'PigFull_Initial.exp',
                'rheology': 'PIG2018.B.tif',
                'degree': 2,
                'friction': 'weertman',
                'maxSteps': 30,
                'rtol': 0.5e-3,
                'noViscosity': False,
                'initWithDeg1': False,
                'plotResult': False,
                'params': None,
                'inversionResult': None,
                'uThresh': 300,  # Here&down nondefault only through params file
                'alpha': 2000
                }
    parser = argparse.ArgumentParser(
        description='\n\n\033[1mRun inversion on Pig \033[0m\n\n')
    parser.add_argument('--geometry', type=str, default=None,
                        help=f'Yaml file with geometry file info '
                        f'[{defaults["geometry"]}] ')
    parser.add_argument('--velocity', type=str, default=None,
                        help=f'Velocity data [{defaults["velocity"]}]')
    parser.add_argument('--mesh', type=str, default=None,
                        help=f'Argus mesh file [{defaults["mesh"]}]')
    parser.add_argument('--rheology', type=str, default=None,
                        help=f'Rheology [defaults["rheology"]')
    parser.add_argument('--degree', type=int, default=None,
                        choices=[1, 2],
                        help=f'Degree for mesh [{defaults["degree"]}]')
    parser.add_argument('--friction', type=str, default=None, 
                        choices=["weertman", "schoof"],
                        help=f'Friction law [{defaults["friction"]}]')
    parser.add_argument('--maxSteps', type=int, default=None,
                        help=f'Max steps for inversion [{defaults["maxSteps"]}]')
    parser.add_argument('--rtol', type=float, default=None,
                        help=f'Convergence tolerance [{defaults["rtol"]}]')
    parser.add_argument('--noViscosity', action='store_true', default=None,
                        help=f'No inversion for shelf viscosity '
                        f'[{defaults["noViscosity"]}]')
    parser.add_argument('--initWithDeg1', action='store_true', default=None,
                        help=f'Initialize deg. 2 with deg. 1 of same name '
                        f'[{defaults["initWithDeg1"]}]')
    parser.add_argument('--plotResult', action='store_true',
                        default=None,
                        help=f'Display results [{defaults["plotResult"]}]')
    parser.add_argument('--params', type=str, default=None,
                        help=f'Input parameter file (.yaml)'
                        f'[{defaults["params"]}]')
    parser.add_argument('inversionResult', type=str, nargs=1,
                        help=f'File with inversion result')
    #
    inversionParams = parseInversionParams(parser, defaults)
    PETSc.Sys.Print('\n\n**** INVERSION PARAMS ****')
    for key in inversionParams:
        PETSc.Sys.Print(f'{key}: {inversionParams[key]}')
    PETSc.Sys.Print('**** END INVERSION PARAMS ****\n')
    #
    return inversionParams


def parseInversionParams(parser, defaults):
    '''
    Parse model params with the following precedence:
    1) Set at command line,
    2) Set in a parameter file,
    3) Default value.
    '''
    args = parser.parse_args()
    # Read file
    inversionParams = mf.readModelParams(args.params, key='inversionParams')  
    for arg in vars(args):
        # If value input through command line, override existing.
        argVal = getattr(args, arg)
        if argVal is not None:
            inversionParams[arg] = argVal
    for key in defaults:
        if key not in inversionParams:
            inversionParams[key] = defaults[key]
    #
    inversionParams['inversionResult'] = inversionParams['inversionResult'][0]
    # Handle conflicts
    if inversionParams['maxSteps'] <= 0 or inversionParams['rtol'] <= 0.:
        myerror(f'maxSteps ({args.maxSteps}) and rtol {args.rtol} must be > 0')
    if inversionParams['degree'] == 1 and inversionParams['initWithDeg1']:
        myerror(f'degree=1 not compatible with initWithDeg1')
    return inversionParams
        

def getRateFactor(rateFile, Q):
    """Read rate factor as B and convert to A
    Parameters
    ----------
    rateFile : str
        File with rate factor data
    Q : firedrake function space
        function space
    Returns
    -------
    Anew : firedrake function
        A Glenns flow law parameter
    """
    Bras = rasterio.open(rateFile)
    B = icepack.interpolate(Bras, Q)
    convFactor = 1e-6 * (86400*365.25)**(-1/3)
    Anew = firedrake.interpolate((B * convFactor)**-3, Q)
    return Anew


def objective(u):
    """Objective function for model runs (generic)
    Parameters
    ----------
    u : firedrake function
        velocity being evaluated
    Returns
    -------
    E : firedrake function
        Objective function x dx
    """
    deltau = u - uObs
    E = 0.5 * ((deltau[0] / sigmaX)**2 + (deltau[1] / sigmaY)**2) * \
        firedrake.dx(mesh)
    return E


def objectiveBeta(u):
    """Objective function for model runs for inverse beta solved only on
    grounded ice regions.
    Parameters
    ---------
    see objectiveFunction
    """
    deltau = u - uObs
    E = 0.5 * groundedG * ((deltau[0] / sigmaX)**2 +
                           (deltau[1] / sigmaY)**2) * firedrake.dx(mesh)
    return E


def objectiveTheta(u):
    """Objective function for model runs for inverse theta solved only on
    floating ice regions.
    Parameters
    ---------
    see objectiveFunction
    """
    deltau = u - uObs
    E = 0.5 * floatingG * ((deltau[0] / sigmaX)**2 +
                           (deltau[1] / sigmaY)**2) * firedrake.dx(mesh)
    return E


def regularizationBeta(beta):
    """Regularization function for beta in friction inversion
    Parameters
    ----------
    beta : firedrake function
        Beta for friction model with C=beta^2
    Returns
    -------
    R: Firedrake function
        Regularization with dx
    """
    Phi = firedrake.Constant(3.)
    L = firedrake.Constant(np.sqrt(1000)*19.5e3)
    R = 0.5 * groundedG * (L / Phi)**2 * \
        firedrake.inner(grad(beta), grad(beta)) * firedrake.dx(mesh)
    return R


def regularizationTheta(theta):
    """Regularization function for theta in fluidity inversion
    Parameters
    ----------
    theta : firedrake function
        theta for fluidity model with C=theta^2
    Returns
    -------
    R: Firedrake function
        Regularization with dx
    """
    Phi = firedrake.Constant(3.)
    L = firedrake.Constant(np.sqrt(1.)*19.5e3)
    R = 0.5 * floatingG * (L / Phi)**2 * \
        firedrake.inner(grad(theta), grad(theta)) * firedrake.dx(mesh)
    return R


def stepInfo(solver):
    """Printer summary information from solver
    Parameters
    ----------
    solver : firedrake solver
        solver function
    """
    E = firedrake.assemble(solver.objective)
    R = firedrake.assemble(solver.regularization)
    area = firedrake.assemble(firedrake.Constant(1) * firedrake.dx(mesh))
    PETSc.Sys.Print(f'{E/area:g}, {R/area:g} '
                    f'{datetime.now().strftime("%H:%M:%S")}  {area:10.3e}')


def betaInit(s, h, speed, V, Q, Q1, grounded, inversionParams):
    """Compute intitial beta using 0.5 taud.
    Parameters
    ----------
    s : firedrake function
        model surface elevation
    h : firedrake function
        model thickness
    speed : firedrake function
        modelled speed
    V : firedrake vector function space
        vector function space
    Q : firedrake function space
        scalar function space
    grounded : firedrake function
        Mask with 1s for grounded 0 for floating.
    """
    if inversionParams['initWithDeg1']:
        checkFile = f'{inversionParams["inversionResult"]}.deg1'
        beta1 = mf.getCheckPointVars(checkFile, 'betaInv', Q1)['betaInv']
        beta = firedrake.interpolate(beta1, Q)
        return beta
    tauD = firedrake.project(-rhoI * g * h * grad(s), V)
    #
    stress = firedrake.sqrt(firedrake.inner(tauD, tauD))
    PETSc.Sys.Print('stress', firedrake.assemble(stress * firedrake.dx(mesh)))
    fraction = firedrake.Constant(0.95)
    U = max_value(speed, 1)
    C = fraction * stress / U**(1/m)
    if inversionParams['friction'] == 'schoof':
        mExp = 1/m + 1
        U0 = inversionParams['uThresh']
        C = C * (m/(m+1)) * (U0**mExp + U**mExp)**(1/(m+1))
    beta = firedrake.interpolate(firedrake.sqrt(C) * grounded, Q)
    return beta


def thetaInit(Ainit, Q, Q1, grounded, floating, inversionParams):
    """Compute intitial theta on the ice shelf (not grounded).
    Parameters
    ----------
    Ainit : firedrake function
        A Glens flow law A
    Q : firedrake function space
        scalar function space
    Q1 : firedrake function space
        1 deg scalar function space used with 'initWithDeg1'
    grounded : firedrake function
        Mask with 1s for grounded 0 for floating.
    floating : firedrake function
        Mask with 1s for floating 0 for grounded.
    Returns
    -------
    theta : firedrake function
        theta for floating ice
    """
    if inversionParams['initWithDeg1']:
        checkFile = f'{inversionParams["inversionResult"]}.deg1'
        Ainit1 = mf.getCheckPointVars(checkFile, 'AInv', Q1)['AInv']
        Ainit = firedrake.interpolate(Ainit1, Q)
    # theta = firedrake.interpolate(firedrake.sqrt(Ainit) * floating, Q)
    theta = firedrake.interpolate(firedrake.ln(Ainit) * floating, Q)
    C = firedrake.Constant(-10)
    # C = firedrake.Constant(0.)
    theta = firedrake.interpolate(theta * floating + C * grounded, Q)
    return theta, Ainit


def defineProblemBeta(beta, model, h, s, u, Anew, theta, grounded, floating,
                      uThresh, opts):
    """Define problem for friction inversion
    """
    problem = icepack.inverse.InverseProblem(
        model=model,
        method=icepack.models.IceStream.diagnostic_solve,
        objective=objectiveBeta,
        regularization=regularizationBeta,
        state_name='u',
        state=u,
        parameter_name='beta',
        parameter=beta,
        model_args={'h': h, 's': s, 'u0': u, 'A': Anew, 'theta': theta,
                    'grounded': grounded, 'floating': floating, 'tol': 1e-6,
                    'uThresh': uThresh},
        dirichlet_ids=opts['dirichlet_ids'],
    )
    return problem


def defineProblemTheta(theta, model, h, s, u, Anew, beta, grounded, floating,
                       uThresh, opts):
    """Define problem for viscosity inversion
    """
    problem = icepack.inverse.InverseProblem(
        model=model,
        method=icepack.models.IceStream.diagnostic_solve,
        objective=objectiveTheta,
        regularization=regularizationTheta,
        state_name='u',
        state=u,
        parameter_name='theta',
        parameter=theta,
        model_args={'h': h, 's': s, 'u0': u, 'A': Anew, 'beta': beta,
                    'grounded': grounded, 'floating': floating, 'tol': 1e-6,
                    'uThresh': uThresh},
        dirichlet_ids=opts['dirichlet_ids'],
    )
    return problem


def saveInversionResult(inversionParams, modelResults, solverBeta, solverTheta,
                        Aorig, grounded, floating, Q, h, s, zb, uObs):
    """Save results to a firedrake dumbcheckpoint file
    """
    outFile = \
        f'{inversionParams["inversionResult"]}.deg{inversionParams["degree"]}'
    theta = solverTheta.parameter
    if not inversionParams['noViscosity']:
        A = icepack.interpolate(grounded * Aorig +
                                floating * firedrake.exp(theta), Q)
        # A = icepack.interpolate(grounded * Aorig + floating * theta**2, Q)
    else:
        A = Aorig
    # Names used in checkpoint file
    varNames = {'uInv': 'uInv', 'betaInv': 'betaInv', 'AInv': 'AInv',
                'groundedInv': 'groundedInv', 'floatingInv': 'floatingInv',
                'hInv': 'hInv', 'sInv': 'sInv', 'zbInv': 'zbInv',
                'uObsInv': 'uObsInv'}
    # Write results
    with firedrake.DumbCheckpoint(outFile, mode=firedrake.FILE_CREATE) as chk:
        # inversion results
        chk.store(solverBeta.state, name=varNames['uInv'])
        betaOut = icepack.interpolate(solverBeta.parameter * grounded, Q)
        chk.store(betaOut, name=varNames['betaInv'])
        chk.store(A, name=varNames['AInv'])
        # Masks
        chk.store(grounded, name=varNames['groundedInv'])
        chk.store(floating, name=varNames['floatingInv'])
        # Original data
        chk.store(h, name=varNames['hInv'])
        chk.store(s, name=varNames['sInv'])
        chk.store(zb, name=varNames['zbInv'])
        chk.store(uObs, name=varNames['uObsInv'] )
    modelResults['end_time'] = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    # dump model params
    outParams = f'{inversionParams["inversionResult"]}.' \
                f'deg{inversionParams["degree"]}.yaml'
    with open(outParams, 'w') as fpYaml:
        myDicts = {'inversionParams': inversionParams,
                   'modelResults': modelResults, 'varNames': varNames}
        yaml.dump(myDicts, fpYaml)


def velocityError(uO, uI, area, message=''):
    """
    Compute and print velocity error
    """
    deltaV = uO - uI
    vError = firedrake.inner(deltaV, deltaV)
    vErrorAvg = np.sqrt(firedrake.assemble(vError * firedrake.dx) / area)
    PETSc.Sys.Print(f'{message} v error {vErrorAvg:10.2f} (m/yr)')
    return vErrorAvg.item()


def parameterInfo(solver, area, message=''):
    """
    Print various statistics
    """
    floating = solver.problem.model_args['floating']
    grounded = solver.problem.model_args['grounded']
    areaFloating = firedrake.assemble(floating * firedrake.dx(mesh))
    areaGrounded = firedrake.assemble(grounded * firedrake.dx(mesh))
    avgFloat = firedrake.assemble(solver.parameter * floating *
                                  firedrake.dx(mesh)) / areaFloating
    avgGrounded = firedrake.assemble(solver.parameter * grounded *
                                     firedrake.dx(mesh)) / areaGrounded
    PETSc.Sys.Print(f'{message} '
                    f'grounded {avgGrounded:9.2e} float {avgFloat:9.2e}')


def runSolvers(solverBeta, solverTheta, modelResults, nSteps=30, rtol=5.e-3,
               solveViscosity=True):
    """ Run joint solvers """
    Jinitial = np.inf
    #
    area = firedrake.assemble(firedrake.Constant(1) * firedrake.dx(mesh))
    # step solvers
    for i in range(0, nSteps):
        #
        # run solver step for beta
        invertTime = datetime.now()
        J = solverBeta._assemble(solverBeta._J)
        PETSc.Sys.Print(f'\nIteration {i} Convergence test {Jinitial - J:10.3e}'
              f' {rtol * Jinitial:10.3e} {invertTime.strftime("%H:%M:%S")}')
        # Check for convergence
        if (Jinitial - J) < rtol * Jinitial:
            break
        Jinitial = J
        # Solver step
        PETSc.Sys.Print('Beta ', end='')
        solverBeta.step()
        betap = solverBeta.parameter
        PETSc.Sys.Print(f' {datetime.now()-invertTime} min/max '
                        f'{betap.dat.data_ro.min():10.3f} '
                        f'{betap.dat.data_ro.max():10.3f}')
        invertTime = datetime.now()
        modelResults[f'Verror_{i:03}'] = \
            velocityError(uObs, solverBeta.state, area, message='Beta')
        parameterInfo(solverBeta, area, message='Beta')
        #
        # run solver step for theta
        if solveViscosity:
            PETSc.Sys.Print('Theta ', end='')
            solverTheta.problem.model_args['beta'].assign(solverBeta.parameter)
            solverTheta.step()
            solverBeta.problem.model_args['theta'].assign(
                solverTheta.parameter)
            thetap = solverTheta.parameter
            PETSc.Sys.Print(f'{datetime.now()-invertTime} min/max '
                            f'{thetap.dat.data_ro.min():10.3f} '
                            f'{thetap.dat.data_ro.max():10.3f}')
            velocityError(uObs, solverTheta.state, area, message='Theta')
            parameterInfo(solverTheta, area, message='Theta')
    PETSc.Sys.Print(f'Done at {datetime.now().strftime("%H:%M:%S")}')


def plotResults(solverBeta, solverTheta, uObs, inversionParams, Q):
    """
    After the inversion is complete plot the inverted parameter and resulting
    velocity error.
    """
    if inversionParams['solveViscosity']:
        figP, axesP = icepack.plot.subplots(1, 2)
    else:
        figP, axesP = icepack.plot.subplots(1, 1)
        axesP = [axesP]
    cBeta = icepack.plot.tricontourf(solverBeta.parameter, axes=axesP[0])
    figP.colorbar(cBeta, ax=axesP[0])
    if inversionParams['solveViscosity']:
        cTheta = icepack.plot.tricontourf(solverTheta.parameter, axes=axesP[1])
        figP.colorbar(cTheta, ax=axesP[1])
    # velocity error
    figU, axesU = icepack.plot.subplots()
    ui = solverBeta.state
    speedError = firedrake.sqrt(firedrake.inner(uObs - ui, uObs - ui))
    speedError = firedrake.interpolate(speedError, Q)
    levels = np.linspace(0, 200, 26)
    cSpeed = icepack.plot.tricontourf(speedError, levels=levels,
                                      extend='max', axes=axesU)
    figU.colorbar(cSpeed, ax=axesU)
    plt.show()


def main():
    # declare globals - fix later
    global sigmaX, sigmaY, uObs, mesh, floatingG, groundedG
    #
    # process command line arags
    inversionParams = setupPigInversionArgs()
    modelResults = {}
    modelResults['begin_time'] = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    PETSc.Sys.Print(inversionParams)
    startTime = datetime.now()
    frictionModels = {'weertman': mf.weertmanFriction,
                      'schoof': mf.schoofFriction}
    frictionLaw = frictionModels[inversionParams['friction']]
    #
    # Read mesh and setup function spaces
    mesh, Q, V, meshOpts = mf.setupMesh(inversionParams['mesh'],
                                        degree=inversionParams['degree'])
    Q1 = None
    if inversionParams['initWithDeg1']:
        Q1 = firedrake.FunctionSpace(mesh, family='CG', degree=1)
    area = firedrake.assemble(firedrake.Constant(1) * firedrake.dx(mesh))
    opts = {}
    opts['dirichlet_ids'] = meshOpts['dirichlet_ids'] # Opts from mesh
    # Input model geometry and velocity
    zb, s, h, floating, grounded = \
        mf.getModelGeometry(inversionParams['geometry'], Q, smooth=True,
                            alpha=inversionParams['alpha'])
    # Set globabl variables for objective function
    floatingG, groundedG = floating, grounded
    uObs, speed, sigmaX, sigmaY = \
        mf.getModelVelocity(inversionParams['velocity'], Q, V,
                            minSigma=5, maxSigma=100)
    Anew = getRateFactor(inversionParams['rheology'], Q)
    PETSc.Sys.Print(f'run time {datetime.now()-startTime}')
    #
    # Initialize diagnostic solve
    beta = betaInit(s, h, speed, V, Q, Q1, grounded, inversionParams)
    theta, Anew = thetaInit(Anew, Q, Q1, grounded, floating, inversionParams)
    #
    # Setup diagnostic solve
    model = icepack.models.IceStream(friction=frictionLaw,
                                     viscosity=mf.viscosity)
    u = model.diagnostic_solve(u0=uObs, h=h, s=s, A=Anew, beta=beta,
                               theta=theta, grounded=grounded,
                               uThresh=inversionParams['uThresh'],
                               floating=floating, **opts)                      
    velocityError(uObs, u, area, message='Initial error')
    PETSc.Sys.Print(f'Time for initial model {datetime.now() - startTime}')
    PETSc.Sys.Print(f'Objective for initial model '
                    f'{firedrake.assemble(objective(u)):10.3e}')
    #
    # Setup problem for beta
    problemBeta = defineProblemBeta(beta, model, h, s, u, Anew, theta, grounded,
                                    floating, inversionParams['uThresh'], opts)
    solverBeta = icepack.inverse.GaussNewtonSolver(problemBeta, stepInfo)
    #
    # Set up problem for theta
    problemTheta = defineProblemTheta(theta, model, h, s, u, Anew, beta, 
                                      grounded, floating, 
                                      inversionParams['uThresh'], opts)
    solverTheta = icepack.inverse.GaussNewtonSolver(problemTheta, stepInfo)
    #
    # step solvers
    runSolvers(solverBeta, solverTheta, modelResults,
               rtol=inversionParams['rtol'], nSteps=inversionParams['maxSteps'],
               solveViscosity=inversionParams['solveViscosity'])
    #
    # Write results to a dumb check point file
    saveInversionResult(inversionParams, modelResults, solverBeta, solverTheta,
                        Anew, grounded, floating, Q, h, s, zb, uObs)
    # Plot results
    if inversionParams['plotResult']:
        plotResults(solverBeta, solverTheta, uObs, inversionParams, Q)


main()
