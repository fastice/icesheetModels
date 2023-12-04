#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
from modelfunc import myerror
import modelfunc as mf
import firedrake
import icepack
from icepack.constants import ice_density as rhoI
from firedrake import max_value
from datetime import datetime
import icepack.inverse
from firedrake import grad
from icepack.constants import gravity as g
from icepack.constants import weertman_sliding_law as m
import matplotlib.pyplot as plt
import icepack.plot
import icepack.models
import sys
#from firedrake import PETSc
import numpy as np
import yaml


Print = print #PETSc.Sys.Print


# ---- Parse Command Line ----


def setupPigInversionArgs():
    ''' Handle command line args'''
    defaults = {'geometry': 'PigGeometry.yaml',
                'velocity':
                '/home/ian/ModelRuns/PIG2018/Data/velocity/pseudo2000/vQ2000',
                'mesh': 'PigFull_Initial.exp',
                'meshOversample': 2,
                'rheology': 'PIG2018.B.tif',
                'degree': 2,
                'friction': 'weertman',
                'maxSteps': 30,
                'rtol': 0.5e-3,
                'GLTaper': 4000,
                'solveViscosity': True,
                'solveBeta': True,
                'solverMethod': 'GaussNewton',
                'initWithDeg1': False,
                'initFile': None,
                'plotResult': False,
                'params': None,
                'inversionResult': None,
                'uThresh': 300,  # Here&down change only through params file
                'alpha': 2000,
                'regTheta': 1.,
                'regBeta': 1.
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
                        help=f'Rheology [{defaults["rheology"]}')
    parser.add_argument('--degree', type=int, default=None,
                        choices=[1, 2],
                        help=f'Degree for mesh [{defaults["degree"]}]')
    parser.add_argument('--GLTaper', default=defaults["GLTaper"], type=float,
                        help=f'GL taper for floating/grounded masks '
                        f'[{defaults["GLTaper"]}]')
    parser.add_argument('--friction', type=str, default=None,
                        choices=["weertman", "schoof"],
                        help=f'Friction law [{defaults["friction"]}]')
    parser.add_argument('--solverMethod', type=str, default=None,
                        choices=["GaussNewton", "BFGS"],
                        help=f'Friction law [{defaults["friction"]}]')
    parser.add_argument('--solverTolerance', type=float, default=1e-6,
                        help=f'Tolerance for solver')
    parser.add_argument('--maxSteps', type=int, default=None,
                        help=f'Max steps for inversion '
                        f'[{defaults["maxSteps"]}]')
    parser.add_argument('--regTheta', type=float, default=None,
                        help=f'Theta regularization scale '
                        f'[{defaults["regTheta"]}]')
    parser.add_argument('--regBeta', type=float, default=None,
                        help=f'Theta regularization scale '
                        f'[{defaults["regBeta"]}]')
    parser.add_argument('--rtol', type=float, default=None,
                        help=f'Convergence tolerance [{defaults["rtol"]}]')
    parser.add_argument('--noViscosity', action='store_true', default=None,
                        help=f'No inversion for shelf viscosity '
                        f'[{not defaults["solveViscosity"]}]')
    parser.add_argument('--noBeta', action='store_true', default=None,
                        help=f'No inversion for beta (basal stress) '
                        f'[{not defaults["solveBeta"]}]')
    parser.add_argument('--initWithDeg1', action='store_true', default=None,
                        help=f'Initialize deg. 2 with deg. 1 of same name '
                        f'[{defaults["initWithDeg1"]}]')
    parser.add_argument('--initFile', type=str, default=None,
                        help=f'Prior inversion file to initialize results '
                        f'[{defaults["initFile"]}]')
    parser.add_argument('--plotResult', action='store_true',
                        default=None,
                        help=f'Display results [{defaults["plotResult"]}]')
    parser.add_argument('--params', type=str, default=None,
                        help=f'Input parameter file (.yaml)'
                        f'[{defaults["params"]}]')
    parser.add_argument('inversionResult', type=str, nargs=1,
                        help='File with inversion result')
    #
    inversionParams = parseInversionParams(parser, defaults)
    Print('\n\n**** INVERSION PARAMS ****')
    for key in inversionParams:
        Print(f'{key}: {inversionParams[key]}')
    Print('**** END INVERSION PARAMS ****\n')
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
    # params that are remapped an negated
    reMap = {'noViscosity': 'solveViscosity', 'noBeta': 'solveBeta'}
    # Read file
    inversionParams = mf.readModelParams(args.params, key='inversionParams')
    for arg in vars(args):
        # If value input through command line, override existing.
        argVal = getattr(args, arg)
        if arg in reMap:
            arg = reMap[arg]
            argVal = not argVal
        if argVal is not None:
            inversionParams[arg] = argVal
    for key in defaults:
        if key not in inversionParams:
            inversionParams[key] = defaults[key]
    #
    inversionParams['inversionResult'] = inversionParams['inversionResult'][0]
    # Handle conflicts
    if inversionParams['maxSteps'] <= 0 or inversionParams['rtol'] <= 0. or \
            inversionParams['solverTolerance'] <= 0:
        myerror(f'maxSteps ({args.maxSteps}) and rtol {args.rtol} must be > 0')
    if inversionParams['degree'] == 1 and inversionParams['initWithDeg1']:
        myerror('degree=1 not compatible with initWithDeg1')
    #
    if inversionParams['initWithDeg1']:
        inversionParams['initFile'] = \
            f'{inversionParams["inversionResult"]}.deg1'
    return inversionParams


# ----- Initialization routines -----


def betaInit(s, h, speed, V, Q, grounded, inversionParams):
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
    # Use a result from prior inversion
    if inversionParams['initFile'] is not None:
        # This will break on the older files
        print(inversionParams['initFile'], 'beta')
        betaTemp = mf.getCheckPointVars(inversionParams['initFile'],
                                        'betaInv', Q)['betaInv']
        beta1 = icepack.interpolate(betaTemp, Q)
        Print(f'Initialized beta with {inversionParams["initFile"]}')
        return beta1
    # No prior result, so use fraction of taud
    tauD = firedrake.project(-rhoI * g * h * grad(s), V)
    #
    stress = firedrake.sqrt(firedrake.inner(tauD, tauD))
    Print('stress', firedrake.assemble(stress * firedrake.dx))
    fraction = firedrake.Constant(0.95)
    U = max_value(speed, 1)
    C = fraction * stress / U**(1/m)
    if inversionParams['friction'] == 'schoof':
        mExp = 1/m + 1
        U0 = firedrake.Constant(inversionParams['uThresh'])
        C = C * (m/(m+1)) * (U0**mExp + U**mExp)**(1/(m+1))
    beta = firedrake.interpolate(firedrake.sqrt(C) * grounded, Q)
    return beta


def thetaInit(Ainit, Q, grounded, floating, inversionParams):
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
    # Now check if there is a file specificed, and if so, init with that
    if inversionParams['initFile'] is not None:
        Print(f'Init. with theta: {inversionParams["initFile"]}')
        # This will break on the older files
        thetaTemp = mf.getCheckPointVars(inversionParams['initFile'],
                                         'thetaInv', Q)['thetaInv']
        thetaInit = icepack.interpolate(thetaTemp, Q)
        return thetaInit
    # No initial theta, so use initial A to init inversion
    Atheta = mf.firedrakeSmooth(Ainit, alpha=1000)
    theta = firedrake.ln(Atheta)
    theta = firedrake.interpolate(theta, Q)
    return theta


def defineProblemBeta(beta, model, uObs, h, s, u, A, theta, grounded, floating,
                      groundedSmooth, floatingSmooth, sigmaX, sigmaY,
                      inversionParams, opts):
    """Define problem for friction inversion
    """
    # Convert params to firedrake constants
    print(inversionParams['solverTolerance'])
    uThresh = firedrake.Constant(inversionParams['uThresh'])
    regBeta = firedrake.Constant(inversionParams['regBeta'])
    # Define problem
    problem = icepack.inverse.InverseProblem(
        model=model,
        objective=makeObjectiveBeta(uObs, grounded, sigmaX, sigmaY),
        regularization=makeRegularizationBeta(grounded, regBeta),
        state_name='velocity',
        state=u,
        parameter_name='beta',
        parameter=beta,
        diagnostic_solve_kwargs={'thickness': h, 'surface': s, 'fluidity': A,
                                 'theta': theta, 'grounded': grounded,
                                 'floating': floating, 'uThresh': uThresh,
                                 'groundedSmooth': groundedSmooth,
                                 'floatingSmooth': floatingSmooth},
        solver_kwargs={**opts, 'tolerance': inversionParams['solverTolerance'],
                       'max_iterations': 203,
                       'diagnostic_solver_parameters': {'snes_max_it': 200,
                                                        'max_iterations': 206}}
    )
    return problem


def defineProblemTheta(theta, model, uObs, h, s, u, A, beta, grounded,
                       floating, groundedSmooth, floatingSmooth,
                       sigmaX, sigmaY, inversionParams, opts):
    """Define problem for viscosity inversion
    """
    # Convert params to firedrake constants
    uThresh = firedrake.Constant(inversionParams['uThresh'])
    regTheta = firedrake.Constant(inversionParams['regTheta'])
    # Define problem
    problem = icepack.inverse.InverseProblem(
        model=model,
        objective=makeObjectiveTheta(uObs, floatingSmooth, sigmaX, sigmaY),
        regularization=makeRegularizationTheta(regTheta),
        state_name='velocity',
        state=u,
        parameter_name='theta',
        parameter=theta,
        diagnostic_solve_kwargs={'thickness': h, 'surface': s, 'fluidity': A,
                                 'beta': beta, 'grounded': grounded,
                                 'floating': floating, 'uThresh': uThresh,
                                 'groundedSmooth': groundedSmooth,
                                 'floatingSmooth': floatingSmooth},
        solver_kwargs={**opts, 'tolerance': inversionParams['solverTolerance'],
                       'max_iterations': 201,
                       'diagnostic_solver_parameters': {'snes_max_it': 200,
                                                        'max_iterations': 205}}
    )
    return problem


def setupSolvers(thickness, surface, velocity, uObs, fluidity, grounded,
                 floating, groundedSmooth, floatingSmooth, sigmaX, sigmaY,
                 beta, theta, model, opts, inversionParams, solverMethod,
                 mesh):
    ''' Set up the solvers '''
    #
    # Set up problem for theta
    solverBeta = None
    if inversionParams['solveBeta']:
        problemBeta = defineProblemBeta(beta, model,  uObs, thickness, surface,
                                        velocity, fluidity, theta,
                                        grounded, floating,
                                        groundedSmooth, floatingSmooth,
                                        sigmaX, sigmaY,
                                        inversionParams, opts)
        solverBeta = solverMethod(problemBeta, makeStepInfo(mesh),
                                  search_max_iterations=500,
                                  search_tolerance=
                                  inversionParams['solverTolerance'])
    #
    # Set up problem for theta
    solverTheta = None
    if inversionParams['solveViscosity']:
        problemTheta = defineProblemTheta(theta, model, uObs, thickness,
                                          surface, velocity, fluidity, beta,
                                          grounded, floating,
                                          groundedSmooth, floatingSmooth,
                                          sigmaX, sigmaY,
                                          inversionParams, opts)
        solverTheta = solverMethod(problemTheta, makeStepInfo(mesh),
                                   search_max_iterations=500,
                                   search_tolerance=
                                   inversionParams['solverTolerance'])
    return solverBeta, solverTheta

# ----- Objective/Regularization Functions


def makeObjectiveBeta(uObs, grounded, sigmaX, sigmaY):
    ''' Allows passing additional variables to objectiveBeta '''
    def objectiveBeta(u):
        """Objective function for model runs for inverse beta solved only on
        grounded ice regions.
        Parameters
        ---------
        see objectiveFunction
        """
        deltau = u - uObs
        E = 0.5 * grounded * ((deltau[0] / sigmaX)**2 +
                              (deltau[1] / sigmaY)**2) * firedrake.dx
        return E
    return objectiveBeta


def makeObjectiveTheta(uObs, floatingSmooth, sigmaX, sigmaY):
    ''' Allows passing additional variables to objectiveTheta'''
    def objectiveTheta(u):
        """Objective function for model runs for inverse theta solved only on
        floating ice regions.
        Parameters
        ---------
        see objectiveFunction
        """
        deltau = u - uObs
        E = 0.5 * floatingSmooth * ((deltau[0] / sigmaX)**2 +
                                    (deltau[1] / sigmaY)**2) * firedrake.dx
        return E
    return objectiveTheta


def makeRegularizationBeta(grounded, regBeta):
    ''' Allows passing additional variables to rebularizationBeta '''
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
        L = firedrake.Constant(np.sqrt(1000.) * 19.5e3)
        R = 0.5 * regBeta * grounded * (L / Phi)**2 * \
            firedrake.inner(grad(beta), grad(beta)) * firedrake.dx
        return R
    return regularizationBeta


def makeRegularizationTheta(regTheta):
    ''' Allows passing additional variables to regularizationTheta '''
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
        L = firedrake.Constant(4 * 19.5e3)
        # floating mask will not stop steep gradient at gl
        R1 = regTheta * 0.5 * (L / Phi)**2 * \
            firedrake.inner(grad(theta), grad(theta))
        R = R1 * firedrake.dx
        return R
    return regularizationTheta


# ----- Viscosity and Friction ------


def taperedViscosity(velocity, thickness, fluidity, theta,
                     groundedSmooth, floatingSmooth):
    ''' This is a test version to use feathered grouning to floating transition
    '''
    # Combine tapered A on grouned and theta for smooth
    A = groundedSmooth * fluidity + floatingSmooth * firedrake.exp(theta)
    viscosity = \
        icepack.models.viscosity.viscosity_depth_averaged(velocity=velocity,
                                                          thickness=thickness,
                                                          fluidity=A)
    return viscosity


def getFrictionLaw(modelName):
    ''' Lookup the friction model by model name (weertman or schoof) '''
    # Available models; augment as needed.
    frictionModels = {'weertman': mf.weertmanFriction,
                      'schoof': mf.schoofFriction}
    return frictionModels[modelName]


def getSolverMethod(solverMethodName):
    ''' Lookup solver method by name (GaussNewton, BFGS) '''
    # available methods; add new methods as needed.
    solverMethods = {'GaussNewton': icepack.inverse.GaussNewtonSolver,
                     'BFGS': icepack.inverse.BFGSSolver}
    return solverMethods[solverMethodName]


def setupTaperedMasks(inversionParams, grounded, floating):
    ''' Smooth or copy floating and grounded masks for tapering near gl
    '''
    # global floatingSmooth, groundedSmooth
    if inversionParams['GLTaper'] < 1:
        floatingSmooth = floating.copy(deepcopy=True)
        groundedSmooth = grounded.copy(deepcopy=True)
    else:
        groundedSmooth = mf.firedrakeSmooth(grounded,
                                            alpha=inversionParams['GLTaper'])
        floatingSmooth = mf.firedrakeSmooth(floating,
                                            alpha=inversionParams['GLTaper'])
    return groundedSmooth, floatingSmooth


# ---- Print messages ----


def makeStepInfo(mesh):
    ''' Allows passing additional variables to stepInfo '''
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
        Print(f'{E/area:g}, {R/area:g} '
              f'{datetime.now().strftime("%H:%M:%S")}  {area:10.3e}')
    return stepInfo


def parameterInfo(solver, area, message=''):
    """
    Print various statistics
    """
    floating = solver.problem.diagnostic_solve_kwargs['floating']
    grounded = solver.problem.diagnostic_solve_kwargs['grounded']
    areaFloating = firedrake.assemble(floating * firedrake.dx)
    areaGrounded = firedrake.assemble(grounded * firedrake.dx)
    avgFloat = firedrake.assemble(solver.parameter * floating *
                                  firedrake.dx) / areaFloating
    avgGrounded = firedrake.assemble(solver.parameter * grounded *
                                     firedrake.dx) / areaGrounded
    Print(f'{message} grounded {avgGrounded:9.2e} floating {avgFloat:9.2e}')


def printExtremes(**kwargs):
    ''' Print min/max of firedrake functions to flag bad inputs'''
    Print('Min/Max of input values')
    Print(''.join(['-']*40))
    for arg in kwargs:
        Print(arg, kwargs[arg].dat.data_ro.min(),
              kwargs[arg].dat.data_ro.max())
    Print(''.join(['-']*40))

# ---- Perform Inversion -----


def solverStep(solver, solverAlt, area, JLast, uObs, solverName, altName,
               rtol):
    """Advance solution a step for beta or theta solver
    """
    invertTime = datetime.now()
    converge = False
    # Update from last step of alternate solver
    if solverAlt is not None:
        solver.problem.diagnostic_solve_kwargs[altName].assign(
            solverAlt.parameter)
    # Solver step
    Print(f'\033[1m{solverName}\033[0m', end='\n')
    # Print(f'GN n iterations {solver._search_solver._iteration} ', end='')
    solver.step()
    # Print(f'-- {solver._search_solver._iteration} ')
    # Compute and print progress info
    parameterInfo(solver, area, message=solverName)
    ve = mf.velocityError(uObs, solver.state, area, message=solverName)
    # test for convergence
    J = solver._assemble(solver._J)
    # info
    Print(f' {datetime.now()-invertTime} min/max '
          f'{solver.parameter.dat.data_ro.min():10.3f} '
          f'{solver.parameter.dat.data_ro.max():10.3f}')
    Print(f'JLast={JLast}, J={J}')
    Print(f'Convergence test {JLast - J:10.3e}'
          f' {rtol * JLast:10.3e} {invertTime.strftime("%H:%M:%S")}')
    # Check for convergence
    if (JLast - J) < rtol * JLast:
        converge = True
    return converge, J, ve


def runSolvers(solverBeta, solverTheta, modelResults, uObs, mesh,
               nSteps=30, rtol=5.e-3, solveViscosity=True, solveBeta=True):
    """ Run joint solvers """
    JLastBeta, JLastTheta = np.inf, np.inf
    #
    area = firedrake.assemble(firedrake.Constant(1) * firedrake.dx(mesh))
    #
    # step solvers
    convergeTheta, convergeBeta = False, False
    for i in range(0, nSteps):
        #
        # run solver step for beta
        Print(f'\n\033[1mIteration {i}\033[0m')
        if solverBeta is not None:
            convergeBeta, JLastBeta, ve = solverStep(solverBeta, solverTheta,
                                                     area, JLastBeta, uObs,
                                                     'beta', 'theta', rtol)
        #
        # run solver step for theta
        if solverTheta is not None:
            convergeTheta, JLastTheta, ve = solverStep(solverTheta, solverBeta,
                                                       area, JLastTheta, uObs,
                                                       'theta', 'beta', rtol)
        modelResults[f'Verror_{i:03}'] = ve  # Last error for step
        if convergeTheta and convergeBeta:  # Done??
            break
    # Done, print message
    Print(f'Done at {datetime.now().strftime("%H:%M:%S")}')


# ---- Plot Results ----

def plotResults(solverBeta, solverTheta, uObs, uFinal, inversionParams, Q):
    """
    After the inversion is complete plot the inverted parameter and resulting
    velocity error.
    """
    if not inversionParams['plotResult']:
        return
    #
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
    figU, axesU = icepack.plot.subplots(1, 2)
    if solverTheta is None:
        ui = solverBeta.state
    else:
        ui = solverTheta.state
    speedError = firedrake.sqrt(firedrake.inner(uObs - ui, uObs - ui))
    speedError = firedrake.interpolate(speedError, Q)
    speedErrorFinal = firedrake.sqrt(firedrake.inner(uFinal - ui, uFinal - ui))
    speedErrorFinal = firedrake.interpolate(speedErrorFinal, Q)
    levels = np.linspace(0, 200, 26)
    cSpeed = icepack.plot.tricontourf(speedError, levels=levels,
                                      extend='max', axes=axesU[0])
    figU.colorbar(cSpeed, ax=axesU[0])
    cSpeedFinal = icepack.plot.tricontourf(speedErrorFinal, levels=levels,
                                           extend='max', axes=axesU[1])
    figU.colorbar(cSpeedFinal, ax=axesU[1])
    plt.show()


def saveInversionResult(mesh, inversionParams, modelResults, solverBeta,
                        solverTheta, A, theta, beta, grounded, floating,
                        h, s, zb, uObs):
    """
    Save results to a firedrake dumbcheckpoint file
    """
    outFile = \
        f'{inversionParams["inversionResult"]}.deg{inversionParams["degree"]}.h5'
    # Names used in checkpoint file - use dict for yaml dump
    varNames = {'uInv': 'uInv', 'betaInv': 'betaInv', 'AInv': 'AInv',
                'groundedInv': 'groundedInv', 'floatingInv': 'floatingInv',
                'hInv': 'hInv', 'sInv': 'sInv', 'zbInv': 'zbInv',
                'uObsInv': 'uObsInv', 'thetaInv': 'thetaInv'}
    # variables to constrain inversion
    myVars = {'AInv': A, 'groundedInv': grounded, 'floatingInv': floating,
              'hInv': h, 'sInv': s, 'zbInv': zb, 'uObsInv': uObs}
    # Write results to check point file
    with firedrake.CheckpointFile(outFile, 'w') as chk:
        chk.save_mesh(mesh)
        # Beta solution
        if solverBeta is not None:  # Save inversion
            chk.save_function(solverBeta.parameter, name=varNames['betaInv'])
        else:  # Save value used
            chk.save_function(beta, name=varNames['betaInv'])
        # Theta solution
        if solverTheta is not None:  # Save param and final state if solved
            chk.save_function(solverTheta.parameter, name=varNames['thetaInv'])
            chk.save_function(solverTheta.state, name=varNames['uInv'])
        else:  # Save theta used throughout model and final result
            chk.save_function(theta, name=varNames['thetaInv'])
            # Save beta v
            chk.save_function(solverBeta.state, name=varNames['uInv'])
        # Save other variables
        for myVar in myVars:
            chk.save_function(myVars[myVar], name=myVar)
    # Save end time
    modelResults['end_time'] = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    # dump inputs and summary data to yaml file
    outParams = f'{inversionParams["inversionResult"]}.' \
                f'deg{inversionParams["degree"]}.yaml'
    with open(outParams, 'w') as fpYaml:
        myDicts = {'inversionParams': inversionParams,
                   'modelResults': modelResults, 'varNames': varNames}
        yaml.dump(myDicts, fpYaml)


# ----- Main ----


def main():
    #
    # process command line arags
    inversionParams = setupPigInversionArgs()
    modelResults = {}
    modelResults['begin_time'] = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    Print(inversionParams)
    startTime = datetime.now()
    # Get friction law and solver methods
    frictionLaw = getFrictionLaw(inversionParams['friction'])
    solverMethod = getSolverMethod(inversionParams['solverMethod'])
    #
    # Read mesh and setup function spaces
    if inversionParams['initFile'] is not None:
        meshI = mf.getMeshFromCheckPoint(inversionParams['initFile'])
    else:
        meshI = None
    mesh, Q, V, meshOpts = \
        mf.setupMesh(inversionParams['mesh'],
                     degree=inversionParams['degree'],
                     meshOversample=inversionParams['meshOversample'],
                     newMesh=meshI)
    Print(f'Mesh Elements={mesh.num_cells()} Vertices={mesh.num_vertices()}')
    # Set up deg 1 function space if using deg 1 to init solution
    #Q1 = None
    #if inversionParams['initWithDeg1']:
    #    Q1 = firedrake.FunctionSpace(mesh, family='CG', degree=1)
    area = firedrake.assemble(firedrake.Constant(1) * firedrake.dx(mesh))
    opts = {'dirichlet_ids': meshOpts['dirichlet_ids']}  # Opts from mesh
    #
    # Input model geometry and velocity
    zb, s, h, floating, grounded = \
        mf.getModelGeometry(inversionParams['geometry'], Q, smooth=True,
                            alpha=inversionParams['alpha'])
    #
    # Smooth versions of masks for tapered function
    groundedSmooth, floatingSmooth = setupTaperedMasks(inversionParams,
                                                       grounded, floating)
    # Get speed
    uObs, speed, sigmaX, sigmaY = \
        mf.getModelVelocity(inversionParams['velocity'], Q, V,
                            minSigma=5, maxSigma=100)
    # Get initial guess for rheology
    A = mf.getRateFactor(inversionParams['rheology'], Q)
    #
    # Initialize beta and theta
    Print(f'run time {datetime.now()-startTime}')
    beta = betaInit(s, h, speed, V, Q, grounded, inversionParams)
    theta = thetaInit(A, Q, grounded, floating, inversionParams)
    # Print min/max for quick QA of inputs
    printExtremes(h=h, s=s, A=A, beta=beta, theta=theta)
    # Assign uThresh here to force correct type for parameter
    uThresh = firedrake.Constant(inversionParams['uThresh'])
    #
    # Setup diagnostic solve and solve for inital model velocity
    model = icepack.models.IceStream(friction=frictionLaw,
                                     viscosity=taperedViscosity)
    solver = icepack.solvers.FlowSolver(model, **opts)
    print(opts)

    # Initial solve
    u = solver.diagnostic_solve(velocity=uObs, thickness=h, surface=s,
                                fluidity=A,
                                beta=beta, theta=theta, grounded=grounded,
                                groundedSmooth=groundedSmooth,
                                floatingSmooth=floatingSmooth,
                                uThresh=uThresh,
                                floating=floating)
    # Compute initial error and objective funtion
    mf.velocityError(uObs, u, area, message='Initial error')
    Print(f'Time for initial model {datetime.now() - startTime}')
    initObj = firedrake.assemble(
        makeObjectiveBeta(uObs, grounded + floating, sigmaX, sigmaY)(u))
    Print(f'Objective for initial model {initObj:10.3e}')
    #
    # Setup problem for beta
    solverBeta, solverTheta = setupSolvers(h, s, u, uObs, A,
                                           grounded, floating,
                                           groundedSmooth, floatingSmooth,
                                           sigmaX, sigmaY,
                                           beta, theta, model, opts,
                                           inversionParams, solverMethod, mesh)
    #
    # step solvers to do the actuall inversion
    runSolvers(solverBeta, solverTheta, modelResults, uObs, mesh,
               rtol=inversionParams['rtol'],
               nSteps=inversionParams['maxSteps'],
               solveViscosity=inversionParams['solveViscosity'],
               solveBeta=inversionParams['solveBeta'])
    sys.stdout.flush()
    #
    # Write results to a dumb check point file
    saveInversionResult(mesh, inversionParams, modelResults, solverBeta,
                        solverTheta, A, theta, beta, grounded, floating, h, s,
                        zb, uObs)
    # Do a final forward solver to evaluate error
    thetaFinal = theta
    if inversionParams['solveViscosity']:
        thetaFinal = solverTheta.parameter
    betaFinal = beta
    if inversionParams['solveBeta']:
        betaFinal = solverBeta.parameter
    # Solution to check that forward model consistent with inverse result
    uFinal = solver.diagnostic_solve(velocity=uObs, thickness=h, surface=s,
                                     fluidity=A, theta=thetaFinal,
                                     beta=betaFinal,
                                     grounded=grounded,
                                     floating=floating,
                                     groundedSmooth=groundedSmooth,
                                     floatingSmooth=floatingSmooth,
                                     uThresh=uThresh)
    mf.velocityError(uObs, uFinal, area, message='Final error')
    # Plot results
    plotResults(solverBeta, solverTheta, uObs, uFinal, inversionParams, Q)


main()
