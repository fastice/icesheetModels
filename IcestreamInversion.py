#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
from icepack.statistics import StatisticsProblem, MaximumProbabilityEstimator
from modelfunc import myerror, printExtremes, taperedViscosityIS
import modelfunc as mf
import firedrake
import icepack
from datetime import datetime
from firedrake import grad
import icepack.plot
import icepack.models
import yaml
from firedrake import PETSc
options = PETSc.Options()
options['options_left'] = False


Print = print  # PETSc.Sys.Print

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
                'maxSteps': 50,
                'rtol': 0.5e-3,
                'GLTaper': 4000,
                'solveViscosity': False,
                'solveBeta': True,
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
    # parser.add_argument('--solverMethod', type=str, default=None,
    #                     choices=["GaussNewton", "BFGS"],
    #                     help=f'Friction law [{defaults["friction"]}]')
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

    if inversionParams['regBeta'] <= 0 or inversionParams['regTheta'] <= 0:
        myerror(f"regBeta {inversionParams['regBeta']} and "
                f"regTheta {inversionParams['regTheta']} must be > 0")
    #
    return inversionParams


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
        thetaInit = firedrake.Function(Q)
        thetaInit.interpolate(thetaTemp)
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


def setupModel(frictionLaw, viscosity, **opts):
    '''
    Setup icestream forward model

    Parameters
    ----------
    frictionLaw : function
        Friction law function.
    viscosity : function
        Viscosity function.
    **opts : dict/keywords
        Data for the forward solution.

    Returns
    -------
    solver : Function
        Forward solver.

    '''
    model = icepack.models.IceStream(friction=frictionLaw,
                                     viscosity=viscosity)
    return icepack.solvers.FlowSolver(model, **opts)


def setupInversion(solver, beta, theta, h, s, A, uObs, grounded,
                   floating, groundedSmooth, floatingSmooth, sigmaX, sigmaY,
                   opts, inversionParams, meshArea):
    '''
    Sets up the functions needed to perform the inversion

    Parameters
    ----------
    solver : TYPE
        The forward solver function.
    beta : TYPE
        The basal shear stress parameter.
    theta : TYPE
        The viscosity parameter for the ice shelf.
    h : TYPE
        Thickness.
    s : TYPE
        Surface topography.
    A : TYPE
        The flow law parameter for grounded areas.
    uObs : firedrake
        Observed velocity..
    grounded : TYPE
        Grounded area mask.
    floating : TYPE
        Floating area mask.
    groundedSmooth : TYPE
        Tapered grounded mask to blend viscosity at grounding line.
    floatingSmooth : TYPE
        Tapered floating mask to blend viscosity at grounding line.
    sigmaX : firedrake
        X component of velocity error.
    sigmaY : firedrake
        X component of velocity error...
    opts : TYPE
        DESCRIPTION.
    inversionParams : TYPE
        DESCRIPTION.
    meshArea : TYPE
        DESCRIPTION.

    Returns
    -------
    function
        Solver to do the inversion.
    '''
    #
    # Make the objective function
    if inversionParams['solveBeta'] and not inversionParams['solveViscosity']:
        lossFunctional = makeObjectiveFunction(uObs, sigmaX, sigmaY, meshArea,
                                               mask=grounded)
        controls = beta
    else:
        lossFunctional = makeObjectiveFunction(uObs, sigmaX, sigmaY, meshArea,
                                               mask=None)
        controls = [beta, theta]
    #
    # Make the regularization function
    regularization = makeRegularization(inversionParams['regBeta'],
                                        inversionParams['regTheta'],
                                        grounded,
                                        floating,
                                        meshArea)
    # Stuff the model params in a dict to pass to model
    simKeywords = {'velocity': uObs,
                   'thickness': h,
                   'surface': s,
                   'fluidity': A,
                   'grounded': grounded,
                   'floating': floating,
                   'groundedSmooth': groundedSmooth,
                   'floatingSmooth': floatingSmooth
                   }
    #
    # Note solving for velocity so use the default
    if not inversionParams['solveViscosity']:
        simKeywords['theta'] = theta
    # Create the simulation
    simulation = defineSimulation(solver, **simKeywords)
    # Run the simulation as test and check the outputs
    u = simulation(controls)
    Print('Sover setup solution')
    printExtremes(u=u)
    # Setup the statistics problem
    myProblem = StatisticsProblem(simulation=simulation,
                                  loss_functional=lossFunctional,
                                  regularization=regularization,
                                  controls=controls)
    # Return the inversion solver
    return \
        MaximumProbabilityEstimator(myProblem,
                                    gradient_tolerance=1e-6,
                                    step_tolerance=1e-1,
                                    max_iterations=inversionParams['maxSteps'])


# ----- Objective/Regularization Functions


def makeObjectiveFunction(uObs, sigmaX, sigmaY, area, mask=None):
    '''
    Create objective function for simulation with observed velocity and errors.

    Parameters
    ----------
    uObs : firedrake
        Observed velocity.
    sigmaX : firedrake
        X component of velocity error.
    sigmaY : firedrake
        X component of velocity error..
    area : firedrake constant
        DESCRIPTION.
    mask : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    function
        Customized objective function.
    '''
    def objectiveFunction(u):
        '''
        Parameters
        ----------
        u : firedrake
            Modeled velocity.

        Returns
        -------
        firedrake
            Objective function.

        '''
        deltau = u - uObs
        if mask is not None:
            return 0.5 * mask/area * ((deltau[0] / sigmaX)**2 +
                                      (deltau[1] / sigmaY)**2) * firedrake.dx
        else:
            return 0.5 / area * ((deltau[0] / sigmaX)**2 +
                                 (deltau[1] / sigmaY)**2) * firedrake.dx
    return objectiveFunction


def makeRegularization(regBeta, regTheta, grounded, floating, meshArea):
    '''
    Creates a regularization function for single or joint inversion
    Parameters
    ----------
    regBeta : float
        regularization constant for beta.
    regTheta : float
        regularization constant for theta.
    grounded : firedrake function
        Mask for grounded area.
    floating : firedrake function
        Mask for floating area.
    meshArea : firedrake constant
        Mesh area use to normalize result.
    Returns
    -------
    regularization
        Customized regularization function.
    '''
    def regularization(controls):
        """Regularization function for beta in friction inversion
        Parameters
        ----------
        controls : firedrake function or list of functions
            Beta for friction model for [Beta, Theta] for joint
        Returns
        -------
        R: Firedrake function
            Regularization with dx
        """
        if type(controls) is not list:
            beta = controls
            theta = None
        else:
            beta, theta = controls
        # scaling constants
        a = firedrake.Constant(10e6)
        b = firedrake.Constant(10e6)
        # get mesh
        mesh = grounded.function_space().mesh()
        RBeta = 0.5 * regBeta * grounded * a * firedrake.inner(grad(beta),
                                                               grad(beta))
        # Return if beta only
        if theta is None:
            return RBeta/meshArea * firedrake.dx(mesh)
        # theta if needed
        RTheta = 0.5 * regTheta * floating * b * firedrake.inner(grad(theta),
                                                                 grad(theta))
        # Return dual value
        return (RBeta + RTheta)/meshArea * firedrake.dx(mesh)
    return regularization



# def getFrictionLaw(s, h, uObs, Q, V,  inversionParams):
#     """Compute intitial beta using 0.95 taud.
#     Parameters
#     ----------
#     s : firedrake function
#         model surface elevation
#     h : firedrake function
#         model thickness
#     speed : firedrake function
#         modelled speed
#     V : firedrake vector function space
#         vector function space
#     Q : firedrake function space
#         scalar function space
#     grounded : firedrake function
#         Mask with 1s for grounded 0 for floating.
#     """
#     m = 3
#     speedObs = icepack.interpolate(firedrake.sqrt(firedrake.inner(uObs,
#                                                                   uObs)), Q)
#     tauD = firedrake.project(-rhoI * g * h * grad(s), V)
#     #
#     stress = firedrake.sqrt(firedrake.inner(tauD, tauD))
#     Print('stress', firedrake.assemble(stress * firedrake.dx))
#     print("Speed min, max:", speedObs.dat.data_ro.min(),
#           speedObs.dat.data_ro.max())
#     fraction = firedrake.Constant(0.95)
#     U = max_value(speedObs, 1)
#     # Schoofie
#     if inversionParams['friction'] == 'schoof':
#         mExp = 1/m + 1
#         U0 = firedrake.Constant(inversionParams['uThresh'])
#         expr = fraction * stress * \
#             ((U0**mExp + U**mExp)**(1./(m+1))) * U**(-1./m)
#         C_0 = firedrake.Function(Q).interpolate(expr)

#         # Create RCF log friction
#         def logFriction(**kwargs):
#             u, beta, grounded = map(kwargs.get,
#                                     ("velocity", "beta", "grounded"))
#             uMag = firedrake.sqrt(firedrake.inner(u, u))
#             return grounded * C_0 * firedrake.exp(beta) * \
#                 ((U0**mExp + uMag**mExp)**(m/(m + 1.)) - U0)
#     # Weertman
#     elif inversionParams['friction'] == 'weertman':
#         expr = fraction * stress / U ** (1 / m)
#         C_0 = firedrake.Function(Q).interpolate(expr)

#         # create Weertman version of logFriction
#         def logFriction(**kwargs):
#             u, beta, grounded = map(kwargs.get,
#                                     ("velocity", "beta", "grounded"))
#             u_0 = firedrake.Constant(1)
#             return grounded * m / (m + 1) * C_0 * firedrake.exp(beta) * \
#                 firedrake.sqrt(firedrake.inner(u, u) + u_0**2) ** (1 / m + 1)
#     else:
#         myerror(f"Invalid friction model: {inversionParams['friction']}")
#     return logFriction


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



def saveInversionResult(mesh, inversionParams, modelResults, beta, theta, uInv,
                        A, grounded, floating, h, s, zb, uObs):
    """
    Save results to a firedrake dumbcheckpoint file
    """
    deg = inversionParams["degree"]
    outFile = f'{inversionParams["inversionResult"]}.deg{deg}.h5'
    # Names used in checkpoint file - use dict for yaml dump
    varNames = {'uInv': 'uInv', 'betaInv': 'betaInv', 'AInv': 'AInv',
                'groundedInv': 'groundedInv', 'floatingInv': 'floatingInv',
                'hInv': 'hInv', 'sInv': 'sInv', 'zbInv': 'zbInv',
                'uObsInv': 'uObsInv', 'thetaInv': 'thetaInv'}
    # variables to constrain inversion
    myVars = {'AInv': A, 'groundedInv': grounded, 'floatingInv': floating,
              'hInv': h, 'sInv': s, 'zbInv': zb, 'uObsInv': uObs}
    # Write results to check point file
    with mf.CheckpointFileNFS(outFile, 'w') as chk:
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
    inversionParams = setupInversionArgs()
    modelResults = {}
    modelResults['begin_time'] = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    Print(inversionParams)
    startTime = datetime.now()
    #
    viscosityLaw = taperedViscosityIS
    # Read mesh and setup function spaces
    if inversionParams['initFile'] is not None:
        meshI = mf.getMeshFromCheckPoint(inversionParams['initFile'])
    else:
        meshI = None
    #
    mesh, Q, V, meshOpts = \
        mf.setupMesh(inversionParams['mesh'],
                     degree=inversionParams['degree'],
                     meshOversample=inversionParams['meshOversample'],
                     newMesh=meshI)
    area = firedrake.assemble(firedrake.Constant(1) * firedrake.dx(mesh))
    Print(f'Mesh Elements={mesh.num_cells()} Vertices={mesh.num_vertices()}')
    #
    if inversionParams['initFile'] is not None:
        Print(f'Initialized beta with {inversionParams["initFile"]}')
        betaTemp = mf.getCheckPointVars(inversionParams['initFile'],
                                        'betaInv', Q)['betaInv']
        beta0 = firedrake.Function(Q)
        beta0.interpolate(betaTemp)
    else:
        print('fresh beta')
        beta0 = firedrake.Function(Q)
    #
    opts = {'dirichlet_ids': meshOpts['dirichlet_ids'],
            "diagnostic_solver_type": "petsc",
            "diagnostic_solver_parameters": {
                  "snes_type": "newtonls",
                  "snes_linesearch_type": "cp",
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
    # Get observed speed and velocity
    uObs, speed, sigmaX, sigmaY = \
        mf.getModelVelocity(inversionParams['velocity'], Q, V,
                            minSigma=5, maxSigma=100)
    # Get initial guess for rheology
    if inversionParams['rateFactorB'] is not None:
        A = mf.getRateFactor(inversionParams['rateFactorB'], Q)
    else:
        A = mf.getRateFactor(inversionParams['rateFactorA'], Q, Ainput=True)
    #
    # Initialize beta and theta
    Print(f'run time {datetime.now()-startTime}')
    frictionLaw = mf.getFrictionLaw(s, h, uObs, Q, V,
                                    inversionParams['friction'],
                                    inversionParams['uThresh'])
    #
    theta0 = thetaInit(A, Q, grounded, floating, inversionParams)
    # Print min/max for quick QA of inputs
    printExtremes(h=h, s=s, A=A, beta=beta0, theta=theta0, speed=speed,
                  sigmaX=sigmaX, sigmaY=sigmaY,
                  ground=grounded, floating=floating,
                  groundedSmooth=groundedSmooth, floatingSmooth=floatingSmooth)
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
    vi = mf.velocityError(uObs, u, area, message='Initial error')
    modelResults['initialError'] = vi
    printExtremes(uObs=uObs, uInitial=u)
    # Compute initial error and objective funtion
    Print(f'Time for initial model {datetime.now() - startTime}')
    #
    betaFinal = beta0.copy(deepcopy=True)
    thetaFinal = theta0.copy(deepcopy=True)
    myEstimator = setupInversion(solver, betaFinal, thetaFinal, h, s, A, uObs,
                                 grounded, floating,
                                 groundedSmooth, floatingSmooth,
                                 sigmaX, sigmaY,
                                 opts, inversionParams, area)
    #
    # Run the single or joint solution
    if inversionParams['solveBeta'] and not inversionParams['solveViscosity']:
        Print('solving for beta only')
        betaFinal = myEstimator.solve()
        thetaFinal = theta0
    else:
        Print('solving for beta and Viscosity')
        betaFinal, thetaFinal = myEstimator.solve()
    #
    # Run the forward model with the final values
    uInv = solver.diagnostic_solve(velocity=uObs, thickness=h, surface=s,
                                   fluidity=A,
                                   beta=betaFinal, theta=thetaFinal,
                                   grounded=grounded,
                                   groundedSmooth=groundedSmooth,
                                   floatingSmooth=floatingSmooth,
                                   floating=floating)
    #
    # Compute summary stats
    # error
    ve = mf.velocityError(uObs, uInv, area, message='Final error')
    modelResults['finalError'] = ve
    #
    # loss
    finalLoss = float(firedrake.assemble(
        myEstimator.problem.loss_functional[0](uInv)))
    print('final loss', finalLoss)
    modelResults['finalLoss'] = finalLoss
    # regularization
    finalRegularization = float(firedrake.assemble(
        myEstimator.problem.regularization[0]([betaFinal, thetaFinal])))
    print('final regularization', finalRegularization)
    modelResults['finalRegularization'] = finalRegularization
    #
    # Write results to a dumb check point file
    saveInversionResult(mesh, inversionParams, modelResults, betaFinal,
                        thetaFinal, uInv,
                        A, grounded, floating, h, s, zb, uObs)



main()
