#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
from icepack.statistics import  StatisticsProblem, MaximumProbabilityEstimator

from modelfunc import myerror
import modelfunc as mf
import firedrake
import icepack
from icepack.constants import ice_density as rhoI
from firedrake import max_value
from datetime import datetime
#import icepack.inverse
from firedrake import grad
from icepack.constants import gravity as g
from icepack.constants import weertman_sliding_law as m
import matplotlib.pyplot as plt
import icepack.plot
import icepack.models
import numpy as np
import yaml
from firedrake import PETSc
options = PETSc.Options()
options['options_left'] = False

Print = print #PETSc.Sys.Print

RS = None
# ---- Parse Command Line ----


def setupInversionArgs():
    ''' Handle command line args'''
    defaults = {'geometry': 'PigGeometry.yaml',
                'velocity':
                '/home/ian/ModelRuns/PIG2018/Data/velocity/pseudo2000/vQ2000',
                'mesh': 'PigFull_Initial.exp',
                'meshOversample': 2,
                'rateFactorB': None,
                'rateFactorA': None,
                'degree': 2,
                'friction': 'weertman',
                'maxSteps': 30,
                'rtol': 0.5e-3,
                'GLTaper': 4000,
                'solveViscosity': False,
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
    parser.add_argument('--rateFactorA', type=str, default=None,
                        help=f'rateFactorA [{defaults["rateFactorA"]}')
    parser.add_argument('--rateFactorB', type=str, default=None,
                        help=f'rateFactorB [{defaults["rateFactorB"]}')
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
                        help='Tolerance for solver')
    parser.add_argument('--maxSteps', type=int, default=None,
                        help='Max steps for inversion '
                        f'[{defaults["maxSteps"]}]')  
    parser.add_argument('--meshOversample', type=int, default=None,
                        help=f'Mesh oversample factor '
                        f'[{defaults["meshOversample"]}]')
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
    if inversionParams['rateFactorA'] is None \
            and inversionParams['rateFactorB'] is None:
        myerror('Rate factor not specified')
    if inversionParams['rateFactorA'] is not None \
          and inversionParams['rateFactorB'] is not None:
        myerror('Rate factors specfied twice: \n'
                f'A={inversionParams["rateFactorA"]}\n'
                f'B={inversionParams["rateFactorB"]}')
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
    print("Speed min, max:", speed.dat.data_ro.min(), speed.dat.data_ro.max())
    fraction = Constant(0.95)
    U = max_value(speed, 1)
    #print('U min, max', firedrake.assemble(U), type(U))
    C = fraction * stress / U**(1/m)
    if inversionParams['friction'] == 'schoof':
        mExp = 1/m + 1
        U0 = Constant(inversionParams['uThresh'])
        C = C * (m/(m+1)) * (U0**mExp + U**mExp)**(1/(m+1))
    print('C = ',C, type(C))
    beta = firedrake.Function(Q)
    beta.interpolate(firedrake.sqrt(C) * grounded)
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
    theta = firedrake.Function(Q)
    theta.interpolate(firedrake.ln(Atheta))
    return theta

def defineSimulation(solver, **kwargs):
    '''
    Define ice stream simulation

    Parameters
    ----------
    solver : icestream solve
        Diagnostic solver.
    **kwargs : dict
        keywords to solver.

    Returns
    -------
    None.

    '''
    def runSimulation(controls):
        if type(controls) is not list:
            return solver.diagnostic_solve(beta=controls, **kwargs)
        else:
            beta, theta = controls
            return solver.diagnostic_solve(beta=beta, theta=theta, **kwargs)
    return runSimulation

def optimizationProblem(simulation, lossFunctional, regularization, controls):
    '''
     Define the optization problem given
    Parameters
    ----------
    simulation : 
        Ice stream solution.
    lossFunctional : 
        loss function.
    regularization:
        regularization function
    controls : 
        controls to be solved for single as scalar, double as list.

    Returns
    -------
    StatisticsProblem

    '''
    return StatisticsProblem(simulation=simulation,
                             loss_functional=lossFunctional,
                             regularization=regularization,
                             controls=controls)

def setupInversion(solver, beta, theta, h, s, A, uObs, grounded,
                 floating, groundedSmooth, floatingSmooth, sigmaX, sigmaY,
                  opts, inversionParams):
    ''' Set up the solvers '''
    uThresh = Constant(inversionParams['uThresh'])
    #
    # Set up problem for theta
    
    if inversionParams['solveBeta'] and not inversionParams['solveViscosity']:
        lossFunctional = makeObjectiveFunction(uObs, sigmaX, sigmaY,
                                               mask=grounded)
        print(type(beta))
        controls = beta
    else:
        lossFunctional = makeObjectiveFunction(uObs, sigmaX, sigmaY, mask=None)
        print(type(beta))
        print(type(theta))
        printExtremes(beta=beta, theta=theta)
        #Smyerror("stop setup")
        controls = [beta, theta]
    #
    regularization = makeRegularization(inversionParams['regBeta'],
                                        inversionParams['regTheta'],
                                        grounded, 
                                        floating)  
    
    simKeywords = {'velocity': uObs,
                   'thickness': h,
                   'surface': s,
                   'fluidity': A,
                   'grounded': grounded,
                   'floating': floating,
                   'groundedSmooth': groundedSmooth,
                   'floatingSmooth': floatingSmooth
                   }
    if not inversionParams['solveViscosity']:
        simKeywords['theta'] = theta
    #     simKeywords['groundedSmooth'] = groundedSmooth
    #     simKeywords['floatingSmooth'] = floatingSmooth
                                 
    simulation = defineSimulation(solver, **simKeywords)
    u = simulation(controls)
    printExtremes(u=u, uObs=uObs, grounded=grounded, sigmaX=sigmaX, sigmaY=sigmaY)
    print('loss', firedrake.assemble(lossFunctional(u)))
    print('beta', firedrake.assemble(regularization([beta, theta])))
    print(controls)
    #print('theta', firedrake.assemble(regularization(theta)))
    printExtremes(theta=theta)
    
    #myerror('stp')

    myProblem = StatisticsProblem(simulation=simulation,
                                  loss_functional=lossFunctional,
                                  regularization=regularization,
                                  controls=controls)
    #
    return MaximumProbabilityEstimator(myProblem,
                                       gradient_tolerance=1e-4,
                                       step_tolerance=1e-1,
                                       max_iterations=50)
    return 
   
def Constant(value):
    constant = firedrake.Function(RS)
    constant.assign(value)
    return constant

# ----- Objective/Regularization Functions



def makeObjectiveFunction(uObs, sigmaX, sigmaY, mask=None):
    ''' Allows passing additional variables to objectiveFunction'''
    def objectiveFunction(u):
        """Objective function for model runs for inverse theta solved only on
        floating ice regions.
        Parameters
        ---------
        see objectiveFunction
        """
        print('start o')
        deltau = u - uObs
        if mask is not None:
            print('end o mask')
            return mask * 0.5 * ((deltau[0] / sigmaX)**2 + 
                                 (deltau[1] / sigmaY)**2) * firedrake.dx
        else:
            print('end o no mask')
            return  0.5 * ((deltau[0] / sigmaX)**2 +\
                           (deltau[1] / sigmaY)**2) * firedrake.dx
    return objectiveFunction

def makeRegularization(regBeta, regTheta, grounded, floating):
    ''' Allows passing additional variables to rebularizationBeta '''
    def regularization(controls):
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
        print('start reg', type(controls), regBeta, regTheta)
        if type(controls) is not list:
            beta = controls
            theta = None
        else:
            beta, theta = controls
        # beta
        PhiBeta = Constant(3.)
        LBeta = Constant(np.sqrt(1000.) * 19.5e3)
        RBeta = regBeta * 0.5 * grounded * (LBeta / PhiBeta)**2 * \
            firedrake.inner(grad(beta), grad(beta)) 
        if theta is None:
            print('end reg no theta')
            return RBeta * firedrake.dx
        # theta 
        PhiTheta = Constant(3.)
        LTheta = Constant(4 * 19.5e3)
        # floating mask will not stop steep gradient at gl
        RTheta = regTheta * 0.5 * (LTheta / PhiTheta)**2 * \
            floating * firedrake.inner(grad(theta), grad(theta))
        #
        print('end reg ')
        return (RBeta + RTheta) * firedrake.dx
    return regularization



# ----- Viscosity and Friction ------


def nonTaperedViscosity(velocity, thickness, fluidity):
    ''' This is a test version to use feathered grouning to floating transition
    '''
    # Combine tapered A on grouned and theta for smooth
    return \
        icepack.models.viscosity.viscosity_depth_averaged(velocity=velocity,
                                                          thickness=thickness,
                                                          fluidity=fluidity)



def taperedViscosity(velocity, thickness, fluidity, theta,
                     groundedSmooth, floatingSmooth):
    ''' This is a test version to use feathered grouning to floating transition
    '''
    # Combine tapered A on grouned and theta for smooth
    print('vis   fix.....')
    # A = groundedSmooth * fluidity + floatingSmooth * firedrake.exp(theta)
    A = firedrake.exp(theta)
    viscosity = \
        icepack.models.viscosity.viscosity_depth_averaged(velocity=velocity,
                                                          thickness=thickness,
                                                          fluidity=A)
    return viscosity


def getFrictionLaw(inversionParams):
    ''' Lookup the friction model by model name (weertman or schoof) '''
    # Available models; augment as needed.
    
    print(type(RS))
    uThresh = Constant(inversionParams['uThresh'])
    def frictionLaw(velocity, grounded, beta):
        if inversionParams['friction'] == 'weertman':
            return mf.weertmanFriction(velocity, grounded, beta, uThresh)
        elif inversionParams['friction'] == 'schoof':
            return mf.schoofFriction(velocity, grounded, beta, uThresh)
        myerror("Invalid friction type {inversionParams['friction']}")
    return frictionLaw


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
        #
        floatingSmooth.dat.data[floatingSmooth.dat.data < 0] = 0
        groundedSmooth.dat.data[groundedSmooth.dat.data < 0] = 0
        floatingSmooth.dat.data[floatingSmooth.dat.data > 1] = 1
        groundedSmooth.dat.data[groundedSmooth.dat.data > 1] = 1
    return groundedSmooth, floatingSmooth


# ---- Print messages ----





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

def setupModel(frictionLaw, viscosity, **opts):
    '''
    Setup icestream model

    Parameters
    ----------
    frictionLaw : TYPE
        DESCRIPTION.
    viscosity : TYPE
        DESCRIPTION.
    **opts : TYPE
        DESCRIPTION.

    Returns
    -------
    solver : TYPE
        DESCRIPTION.

    '''
    model = icepack.models.IceStream(friction=frictionLaw,
                                     viscosity=viscosity)
    solver = icepack.solvers.FlowSolver(model, **opts)
    #
    return solver




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
    speedError = icepack.interpolate(speedError, Q)
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


def saveInversionResult(mesh, inversionParams, modelResults, beta, theta, uInv,
                      A, grounded, floating, h, s, zb, uObs):
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
        # Beta solutio
        chk.save_function(beta, name=varNames['betaInv'])
        # Theta solution  or original if not solved for
        chk.save_function(theta, name=varNames['thetaInv'])
        chk.save_function(uInv, name=varNames['uInv'])
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
    global RS
    inversionParams = setupInversionArgs()
 
    modelResults = {}
    modelResults['begin_time'] = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    Print(inversionParams)
    startTime = datetime.now()
  
    #solverMethod = getSolverMethod(inversionParams['solverMethod'])
    viscosityLaw = taperedViscosity
    #viscosityLaw = nonTaperedViscosity
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
    RS = firedrake.FunctionSpace(mesh, "R", 0)
    Print(f'Mesh Elements={mesh.num_cells()} Vertices={mesh.num_vertices()}')
    # Get friction law and solver methods
    frictionLaw = getFrictionLaw(inversionParams)
    # Set up deg 1 function space if using deg 1 to init solution
    #Q1 = None
    #if inversionParams['initWithDeg1']:
    #    Q1 = firedrake.FunctionSpace(mesh, family='CG', degree=1)
    area = firedrake.assemble(Constant(1) * firedrake.dx(mesh))
    opts = {'dirichlet_ids': meshOpts['dirichlet_ids'],
            "diagnostic_solver_type": "petsc",
            "diagnostic_solver_parameters": {
                #"snes_monitor": None,
                "snes_type": "newtonls",
                "snes_max_it":200,
                "snes_linesearch_type": "nleqerr",
                "ksp_type": "gmres",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps",
            }
        }
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
    #print(inversionParams['velocity'])
    #print(inversionParams['meshOversample'])
    #print('A', speed.dat.data_ro.min(), speed.dat.data_ro.max())
    #myerror("STO")
    # Get initial guess for rheology
    if inversionParams['rateFactorB'] is not None:
        A = mf.getRateFactor(inversionParams['rateFactorB'], Q)
    else:
        A = mf.getRateFactor(inversionParams['rateFactorA'], Q, Ainput=True) 
    #
    # Initialize beta and theta
    Print(f'run time {datetime.now()-startTime}')
    beta0 = betaInit(s, h, speed, V, Q, grounded, inversionParams)
    theta0 = thetaInit(A, Q, grounded, floating, inversionParams)
    # Print min/max for quick QA of inputs
    printExtremes(h=h, s=s, A=A, beta=beta0, theta=theta0, speed=speed,
                  ground=grounded, floating=floating,
                  groundedSmooth=groundedSmooth,floatingSmooth=floatingSmooth)
    # Assign uThresh here to force correct type for parameter
    uThresh = Constant(inversionParams['uThresh'])
    #
    solver = setupModel(frictionLaw, viscosityLaw,  **opts)
    #
    # Initial solve
    u = solver.diagnostic_solve(velocity=uObs, thickness=h, surface=s,
                                fluidity=A,
                                beta=beta0, theta=theta0, grounded=grounded,
                                groundedSmooth=groundedSmooth,
                                floatingSmooth=floatingSmooth,
                                floating=floating)
    mf.velocityError(uObs, u, area, message='Initial error')
    #myerror("stop after diag")
    # Compute initial error and objective funtion
   
    Print(f'Time for initial model {datetime.now() - startTime}')
    #
    myEstimator = setupInversion(solver, beta0, theta0, h, s, A, uObs, grounded,
                                 floating, groundedSmooth, floatingSmooth,
                                 sigmaX, sigmaY,
                                 opts, inversionParams)
    from firedrake.adjoint import get_working_tape

    tape = get_working_tape()
    print("Tape state:", tape)

    if inversionParams['solveBeta'] and not inversionParams['solveViscosity']:
        Print('solving for beta only')
        betaFinal =  myEstimator.solve()
        thetaFinal = theta0
    else:
        Print('solving for beta and Viscosity')
        betaFinal, thetaFinal =  myEstimator.solve()
    #
    uInv = solver.diagnostic_solve(velocity=uObs, thickness=h, surface=s,
                                fluidity=A,
                                beta=betaFinal, theta=thetaFinal, 
                                grounded=grounded,
                                groundedSmooth=groundedSmooth,
                                floatingSmooth=floatingSmooth,
                                floating=floating)
    #myerror("stop after diag")
    
    #
    # # Write results to a dumb check point file
    saveInversionResult(mesh, inversionParams, modelResults, betaFinal, 
                        thetaFinal, uInv,
                        A, grounded, floating, h, s, zb, uObs)
        
    mf.velocityError(uObs, uInv, area, message='Final error')
    # # Do a final forward solver to evaluate error
    # thetaFinal = theta
    # if inversionParams['solveViscosity']:
    #     thetaFinal = solverTheta.parameter
    # betaFinal = beta
    # if inversionParams['solveBeta']:
    #     betaFinal = solverBeta.parameter
    # # Solution to check that forward model consistent with inverse result
    # uFinal = solver.diagnostic_solve(velocity=uObs, thickness=h, surface=s,
    #                                  fluidity=A, theta=thetaFinal,
    #                                  beta=betaFinal,
    #                                  grounded=grounded,
    #                                  floating=floating,
    #                                  groundedSmooth=groundedSmooth,
    #                                  floatingSmooth=floatingSmooth,
    #                                  uThresh=uThresh)
    # mf.velocityError(uObs, uFinal, area, message='Final error')
    # # Plot results
    # plotResults(solverBeta, solverTheta, uObs, uFinal, inversionParams, Q)
    #myerror('stop')
    # initObj = firedrake.assemble(
    #     makeObjectiveBeta(uObs, grounded + floating, sigmaX, sigmaY)(u))
    # Print(f'Objective for initial model {initObj:10.3e}')
    # #
    # # Setup problem for beta
    # solverBeta, solverTheta = setupSolvers(h, s, u, uObs, A,
    #                                        grounded, floating,
    #                                        groundedSmooth, floatingSmooth,
    #                                        sigmaX, sigmaY,
    #                                        beta, theta, model, opts,
    #                                        inversionParams, solverMethod, mesh)
    #
    # step solvers to do the actuall inversion
    # runSolvers(solverBeta, solverTheta, modelResults, uObs, mesh,
    #            rtol=inversionParams['rtol'],
    #            nSteps=inversionParams['maxSteps'],
    #            solveViscosity=inversionParams['solveViscosity'],
    #            solveBeta=inversionParams['solveBeta'])
    # sys.stdout.flush()
    # #
    # # Write results to a dumb check point file
    # saveInversionResult(mesh, inversionParams, modelResults, solverBeta,
    #                     solverTheta, A, theta, beta, grounded, floating, h, s,
    #                     zb, uObs)
    # # Do a final forward solver to evaluate error
    # thetaFinal = theta
    # if inversionParams['solveViscosity']:
    #     thetaFinal = solverTheta.parameter
    # betaFinal = beta
    # if inversionParams['solveBeta']:
    #     betaFinal = solverBeta.parameter
    # # Solution to check that forward model consistent with inverse result
    # uFinal = solver.diagnostic_solve(velocity=uObs, thickness=h, surface=s,
    #                                  fluidity=A, theta=thetaFinal,
    #                                  beta=betaFinal,
    #                                  grounded=grounded,
    #                                  floating=floating,
    #                                  groundedSmooth=groundedSmooth,
    #                                  floatingSmooth=floatingSmooth,
    #                                  uThresh=uThresh)
    # mf.velocityError(uObs, uFinal, area, message='Final error')
    # # Plot results
    # plotResults(solverBeta, solverTheta, uObs, uFinal, inversionParams, Q)


main()
