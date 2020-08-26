#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
from utilities import myerror
import modelfunc as mf
import firedrake
import icepack
from icepack.constants import ice_density as rhoI
from datetime import datetime
import icepack.inverse
import matplotlib.pyplot as plt
from matplotlib import cm
import icepack.plot
import icepack.models
from firedrake import PETSc
import numpy as np
import yaml
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable

rhoW = rhoI * 1028./917.  # This ensures rhoW based on 1028

floatingG, groundedG, mesh = None, None, None


def parsePigForwardArgs():
    ''' Handle command line args'''
    defaults = {'geometry': 'PigGeometry.yaml',
                'degree': 1,
                'plotResult': False,
                'prognosticOnly': False,
                'params': None,
                'inversionResult': None,
                'nYears': 10.,
                'GLThresh': 40.,
                'SMB': '/home/ian/ModelRuns/Thwaites/BrookesMap/'
                'OLS_Trend_plus_Resid_9b9.tif',
                'deltaT': 0.05,
                'meltParamsFile': 'meltParams.yaml',
                'meltParams': 'linear',
                'meltModel': 'piecewiseWithDepth',
                'profileFile': '/Volumes/UsersIan/ModelExperiments/'
                'PigForward/llprof/piglong.xy',
                'mapPlotLimits': {'xmin': -1.66e6, 'xmax': -1.51e6,
                                  'ymin': -3.50e5, 'ymax': -2.30e5}
                }
    parser = argparse.ArgumentParser(
        description='\n\n\033[1mRun a forward simulation initialized by an '
        'inversion \033[0m\n\n')
    parser.add_argument('--geometry', type=str, default=None,
                        help=f'Yaml file with geometry file info '
                        f'[{defaults["geometry"]}] ')
    parser.add_argument('--degree', type=int, default=None,
                        choices=[1, 2], help='Degree for mesh ')
    parser.add_argument('--nYears', type=float, default=None,
                        help=f'Simulation length (yrs) [{defaults["nYears"]}]')
    parser.add_argument('--GLThresh', type=float, default=None,
                        help='Threshhold for GL weakening '
                        f'[{defaults["GLThresh"]}]')
    parser.add_argument('--meltParams', type=str, default=None,
                        help='Name of melt params from meltParams.yaml file '
                        f'[{defaults["meltParams"]}]')
    parser.add_argument('--meltParamsFile', type=str, default=None,
                        help='Yaml file with melt params'
                        f'[{defaults["meltParamsFile"]}]')
    parser.add_argument('--deltaT', type=float, default=None,
                        help=f'Time step (yrs) [{defaults["deltaT"]}]')
    parser.add_argument('--plotResult', action='store_true',
                        default=defaults["plotResult"],
                        help=f'Display results [{defaults["plotResult"]}]')
    parser.add_argument('--prognosticOnly', action='store_true', default=False,
                        help=f'Solve prognostic with input vel [{False}]')
    parser.add_argument('--params', type=str, default=None,
                        help=f'Input parameter file (.yaml)'
                        f'[{defaults["params"]}]')
    parser.add_argument('inversionResult', type=str, nargs=1,
                        help='Base name(.degX.yaml/.h5) for inversion result')
    parser.add_argument('forwardResult', type=str, nargs=1,
                        help='Base name forward output')
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
    # Set results initially from params file
    forwardParams = mf.readModelParams(args.params, key='forwardParams')
    # Overwrite with command line
    for arg in vars(args):
        # If value input through command line, override existing.
        argVal = getattr(args, arg)
        if argVal is not None:
            forwardParams[arg] = argVal
    # If not already in params, then use default value
    for key in defaults:
        if key not in forwardParams:
            forwardParams[key] = defaults[key]
    # get rid of lists for main args
    forwardParams['inversionResult'] = \
        f'{forwardParams["inversionResult"][0]}.deg{forwardParams["degree"]}'
    forwardParams['forwardResultDir'] = forwardParams['forwardResult'][0]
    forwardParams['forwardResult'] = forwardParams['forwardResult'][0]
    # append deg x to all ouput files
    forwardParams['forwardResult'] += f'.deg{forwardParams["degree"]}'
    # read inversonParams
    inversionYaml = f'{forwardParams["inversionResult"]}.yaml'
    inversionParams = mf.readModelParams(inversionYaml, key='inversionParams')
    #
    # Grap inversion params for forward sim
    for key in ['friction', 'degree', 'mesh', 'uThresh', 'GLTaper']:
        try:
            forwardParams[key] = inversionParams[key]
        except Exception:
            myerror(f'parseForwardParams: parameter- {key} - missing from'
                    ' inversion result')
    return forwardParams, inversionParams


def viscosityNoTheta(velocity, thickness, grounded, A):
    return icepack.models.viscosity.viscosity_depth_averaged(velocity,
                                                             thickness, A)


def velocityError(uO, uI, area, message=''):
    """
    Compute and print velocity error
    """
    deltaV = uO - uI
    vError = firedrake.inner(deltaV, deltaV)
    vErrorAvg = np.sqrt(firedrake.assemble(vError * firedrake.dx) / area)
    PETSc.Sys.Print(f'{message} v error {vErrorAvg:10.2f} (m/yr)\n')
    return vErrorAvg.item()


def setupTimesSeriesPlots(plotData, forwardParams):
    ''' Setup axes for times series plots
    '''
    figTS, axesTS = plt.subplots(3, 1, figsize=(5.5, 12))
    yLabels = ['dA (km^2)', 'dV (km^3/yr)', 'Tot. Vol. Change (km^3)']
    # axesTS = np.append(axesTS, axesTS[1].twinx())
    for ax, yLabel in zip(axesTS, yLabels):
        ax.set_xlim([0, forwardParams['nYears']])
        ax.set_xlabel('t (years)')
        ax.set_ylabel(yLabel)
    plt.tight_layout()
    plotData['figTS'], plotData['axesTS'] = figTS, axesTS


def volumeChange(grounded, floating, h, hLast, DVF, DVG, dT, mesh):
    ''' Compute volume change on grounded and floating ice. Summing deltaVG
    should automatically provide VAF since the loss is always grounded.
    '''
    deltaVG = firedrake.assemble(grounded * (h - hLast) * firedrake.dx(mesh))
    deltaVF = firedrake.assemble(floating * (h - hLast) * firedrake.dx(mesh))
    return deltaVF/dT, deltaVG/dT, DVF + deltaVF, DVG + deltaVG


def timeSeriesPlots(year, plotData, summaryData):
    ''' Plot time series data
    '''
    if year % summaryData['dTsum'] > .0001:
        return
    t = summaryData['year'][-1]
    # Plot floating and grounded area
    axesTS = plotData['axesTS']
    dShelfArea = (summaryData['fArea'][-1] - summaryData['fArea'][0]) / 1e6
    axesTS[0].plot(t, dShelfArea, 'b.', label='dA floating')
    # Compute and plot volume changes
    # axesTS[2].plot(t, summaryData['Umax-U0'][-1], 'ro', label='Max speedup')
    axesTS[1].plot(t, summaryData['deltaVF'][-1]/1e9, 'bo',
                   label='dV floating')
    axesTS[1].plot(t, summaryData['deltaVG'][-1]/1e9, 'ko',
                   label='dV grounded')
    axesTS[1].plot(t, summaryData['meltTot'][-1]/1e9, 'ro',
                   label='Melt')
    # Do legend on first plot
    if len(summaryData['year']) < 2:
        axesTS[2].plot(np.nan, 'cd', label='Total loss floating')  # lgnd hack
        axesTS[2].plot(np.nan, 'md', label='Total loss grounded')
        # axesTS[1].plot(np.nan, 'md', label='Total loss grounded')
        for ax in axesTS:
            ax.legend()
    # Plot total loss (after others to allow legend hack)
    axesTS[2].plot(t, summaryData['DVF'][-1]/1e9, 'cd', label='Floating loss')
    axesTS[2].plot(t, summaryData['DVG'][-1]/1e9, 'md', label='Grounded loss')
    if plotData['plotResult']:
        plt.draw()
        plt.pause(1.)
        plotData['figTS'].canvas.draw()


def computeSummaryData(SD, h, hLast, s, u, melt, grounded, floating, year, Q,
                       mesh, beginTime):
    ''' Compute summary results and sort in summaryData
        Save all as float for yaml output
    '''
    if year % SD['dTsum'] > .0001 * year or year < SD['dTsum']:
        return False
    #
    print(f'year {year} runtime {datetime.now() - beginTime}')
    #
    gArea = firedrake.assemble(grounded * firedrake.dx(mesh))
    SD['year'].append(float(year))
    SD['gArea'].append(float(gArea))
    SD['fArea'].append(float(SD['area'] - gArea))
    deltaVF, deltaVG, DVF, DVG = \
        volumeChange(grounded, floating, h, hLast, SD['DVF'][-1],
                     SD['DVG'][-1], SD['dTsum'], mesh)
    #
    SD['deltaVF'].append(float(deltaVF))
    SD['deltaVG'].append(float(deltaVG))
    SD['DVF'].append(float(DVF))
    SD['DVG'].append(float(DVG))
    #
    Umax = float(u.dat.data_ro.max())
    SD['Umax-U0'].append(float(Umax - SD['U0max']))
    #
    meltTot = firedrake.assemble(icepack.interpolate(floating * melt, Q) *
                                 firedrake.dx(mesh))
    SD['meltTot'].append(float(meltTot))
    #
    print(f'{year}: Initial melt {SD["meltTot"][0] / 1e9:.2f} current melt '
          f' {meltTot / 1e9:.2f}')
    return True


def setupMapPlots(plotData, first=False):
    ''' Setup plot for map view of melt, grounded/floating, thinning.
    Call for each plot to clear memory
    '''
    # First call just set to None
    if first:
        plotData['figM'], plotData['axesM'] = None, None
        return
    # Close to avoid memory leak
    if plotData['figM'] is not None:
        plt.close(plotData['figM'])
    plotData['figM'], plotData['axesM'] = \
        icepack.plot.subplots(1, 3, figsize=(14, 4))
    lim = plotData['mapPlotLimits']
    for ax in plotData['axesM'].flatten():
        ax.set_xlim(lim['xmin'], lim['xmax'])
        ax.set_ylim(lim['ymin'], lim['ymax'])
    plotData['figMColorbar'] = True


def setupProfilePlots(plotData, forwardParams, h):
    ''' Setup axes for times series plots
    '''
    # If profile doesn't exist, return
    if not os.path.exists(forwardParams['profileFile']):
        plotData['figP'], plotData['axesP'] = None, None
        return
    #
    plotData['profXY'], plotData['distance'] = \
        mf.readProfile(forwardParams['profileFile'], h)
    # Setup fig and axex
    figP, axesP = plt.subplots(4, 1, figsize=(5.5, 12))
    yLabels = ['Speed (m/yr)', 'Elevation (m)', 'melt (m/yr)', 'dh/dt (m/yr)']
    for ax, yLabel in zip(axesP, yLabels):
        ax.set_xlabel('distance (km)')
        ax.set_ylabel(yLabel)
    figP.tight_layout()
    plotData['figP'], plotData['axesP'] = figP, axesP
    # Setup up number of plots
    nPlots = min(20, int(forwardParams['nYears']))
    nPlots = 20
    plotData['profileDT'] = forwardParams['nYears']/nPlots
    print('profile DT ', plotData['profileDT'])
    plotData['nextProfPlotTime'] = -plotData['profileDT']
    plotData['profNum'] = 0
    plotData['cmap'] = cm.get_cmap('hsv', nPlots)(range(nPlots+1))
    plotData['lasth'] = np.array([h((xy[0], xy[1]))
                                 for xy in plotData['profXY']])
    axesP[3].set_ylim(-10, 10)


def profilePlots(t, plotData, u, s, h, zb, zF, melt, Q, first=False):
    '''Plot profiles every nYears/20
    '''
    # Return if not ready for plot
    if t < plotData['nextProfPlotTime'] or plotData['axesP'] is None \
            and not first:
        return
    # Increment time for next plot: use max to kill initial -1
    plotData['nextProfPlotTime'] = max(plotData['nextProfPlotTime'], 0.)
    plotData['nextProfPlotTime'] += plotData['profileDT']
    myColor = plotData['cmap'][plotData['profNum']]
    myLabel = f'year={t:.1f}'
    # First is special case
    if first:
        myColor = 'k'
        zbProf = [zb((xy[0], xy[1])) for xy in plotData['profXY']]
        plotData['axesP'][1].plot(plotData['distance'], zbProf, color=myColor)
        myLabel = 'initial'
        zFProf = [zF((xy[0], xy[1])) for xy in plotData['profXY']]
        plotData['axesP'][1].plot(plotData['distance'], zFProf, color='b')
        # print(zbProf)
    # Now do plots
    speed = firedrake.interpolate(firedrake.sqrt(firedrake.inner(u, u)), Q)
    speedProf = [speed((xy[0], xy[1])) for xy in plotData['profXY']]
    plotData['axesP'][0].plot(plotData['distance'], speedProf,
                              label=myLabel, color=myColor)
    plotData['axesP'][0].legend(ncol=2)
    # Geometry
    sProf = [s((xy[0], xy[1])) for xy in plotData['profXY']]
    bProf = [s((xy[0], xy[1])) - h((xy[0], xy[1]))
             for xy in plotData['profXY']]
    zbProf = [zb((xy[0], xy[1])) for xy in plotData['profXY']]
    plotData['axesP'][1].plot(plotData['distance'], sProf, color=myColor)
    plotData['axesP'][1].plot(plotData['distance'], bProf, color=myColor)
    plotData['axesP'][1].plot(plotData['distance'], zbProf, color='k')
    #
    if melt is not None:
        meltProf = [melt((xy[0], xy[1])) for xy in plotData['profXY']]
        plotData['axesP'][2].plot(plotData['distance'], meltProf,
                                  label=myLabel, color=myColor)
        hProf = np.array([h((xy[0], xy[1])) for xy in plotData['profXY']])
        dhdt = (hProf - plotData['lasth']) / plotData['profileDT']
        plotData['axesP'][3].plot(plotData['distance'], dhdt,
                                  label=myLabel, color=myColor)
        plotData['lasth'] = hProf
    plotData['profNum'] += 1
    # plt.draw()
    if plotData['plotResult']:
        plotData['figP'].canvas.draw()


def initialState(h0, s0, u0, zb, grounded0, floating0, Q):
    '''Make copies of original data to start inversion
    '''
    h, s = h0.copy(deepcopy=True), s0.copy(deepcopy=True)
    hLast = h0.copy(deepcopy=True)
    u = u0.copy(deepcopy=True)
    zF = mf.flotationHeight(zb, Q)
    grounded = grounded0.copy(deepcopy=True)
    floating = floating0.copy(deepcopy=True)
    return h, hLast, s, u, zF, grounded, floating


def initSummary(grounded0, floating0, h0, u0, meltModel, meltParams, Q, mesh):
    ''' Compute intiplt.paual areas and summary data
    '''
    area = firedrake.assemble(firedrake.Constant(1) * firedrake.dx(mesh))
    gArea0 = firedrake.assemble(grounded0 * firedrake.dx(mesh))
    fArea0 = area - gArea0
    Umax0 = u0.dat.data_ro.max()
    melt = meltModel(h0, floating0, meltParams, Q, u0)
    meltTot = firedrake.assemble(icepack.interpolate(floating0 * melt, Q) *
                                 firedrake.dx(mesh))
    summaryData = {'year': [0], 'DVG': [0], 'DVF': [0], 'deltaVF': [0],
                   'deltaVG': [0], 'gArea': [float(gArea0)],
                   'fArea': [float(fArea0)], 'meltTot': [float(meltTot)],
                   'area': float(area), 'U0max': float(Umax0), 'Umax-U0': [0],
                   'dTsum': 1.}
    return summaryData


def setupFriction(forwardParams):
    ''' Error check and return friction law specified by forwardParams
    '''
    try:
        frictionLaw = {'weertman': mf.weertmanFriction,
                       'schoof': mf.schoofFriction}[forwardParams['friction']]
    except Exception:
        myerror(f'setupFriction: Invalid friction law: '
                f'{forwardParams["friction"]}')
    return frictionLaw


def computeSurface(h, zb, Q):
    '''Hack of icepack version to uses different rhoI/rhoW
    '''
    s = firedrake.max_value(h + zb, h * (1 - rhoI/rhoW))
    return icepack.interpolate(s, Q)


def setupMelt(forwardParams):
    '''Parse melt params file and return melt params and model
    '''
    allMeltParams = mf.inputMeltParams(forwardParams['meltParamsFile'])
    try:
        meltParams = allMeltParams[forwardParams['meltParams']]
    except Exception:
        myerror(f'setupMelt: Key error for {forwardParams["meltModel"]} from '
                f'melt params file {forwardParams["meltParamsFile"]}')
    meltModels = {'piecewiseWithDepth': mf.piecewiseWithDepth,
                  'divMelt': mf.divMelt}
    try:
        meltModel = meltModels[forwardParams['meltModel']]
    except Exception:
        myerror(f'setupMelt: Invalid model selection '
                f'{forwardParams["meltModel"]} not in melt def.: {meltModels}')
    return meltModel, meltParams


def setupOutputs(forwardParams, inversionParams, check=True):
    ''' Make output dir and dump forward and inversionParams
    '''
    if not os.path.exists(forwardParams['forwardResultDir']):
        os.mkdir(forwardParams['forwardResultDir'])
    inputsFile = f'{forwardParams["forwardResultDir"]}/' \
                 f'{forwardParams["forwardResult"]}.inputs.yaml'
    print(f'Writing inputs to: {inputsFile}')
    #
    with open(inputsFile, 'w') as fpYaml:
        myDicts = {'forwardParams': forwardParams,
                   'inversionParams': inversionParams}
        yaml.dump(myDicts, fpYaml)
    # open check point file
    chkFile = f'{forwardParams["forwardResultDir"]}/' \
              f'{forwardParams["forwardResult"]}.history'
    if check:
        return firedrake.DumbCheckpoint(chkFile, mode=firedrake.FILE_CREATE)
    return None


def saveSummaryData(forwardParams, summaryData):
    ''' Write summary data to yaml file
    '''
    summaryFile = f'{forwardParams["forwardResultDir"]}/' \
                  f'{forwardParams["forwardResult"]}.summary.yaml'
    # Convert numpy to list
    for s in summaryData:
        if isinstance(summaryData[s], np.ndarray):
            summaryData[s] = summaryData[s].tolist()
    # Now write result to yaml
    with open(summaryFile, 'w') as fpYaml:
        yaml.dump({'summaryFile': summaryData}, fpYaml)


def outputTimeStep(t, chk,  **kwargs):
    ''' Ouput variables at a time step)
    '''
    chk.set_timestep(t)
    for k in kwargs:
        chk.store(kwargs[k], name=k)


def readSMB(SMBfile, Q):
    ''' Read SMB file an limit values to +/- 6 to avoid no data values
    '''
    if not os.path.exists:
        myerror(f'readSMB: SMB file  ({SMBfile}) does not exist')
    SMB = mf.getModelVarFromTiff(SMBfile, Q)
    # avoid any unreasonably large value
    SMB = icepack.interpolate(
        firedrake.max_value(firedrake.min_value(SMB * 1./0.917, 6), -6), Q)
    return SMB


def myColorBar(fig, ax, contours, label=''):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(contours, cax=cax, label=label)


def mapPlots(plotData, t, melt, floating, h, hLast, Q):
    ''' Plot melt, floating/grounded, thinning in map view
    '''
    setupMapPlots(plotData)
    axes = plotData['axesM'].flatten()
    meltC = icepack.plot.tricontourf(melt, levels=np.linspace(-150, 0, 151),
                                     extend='both', axes=axes[0])
    floatC = icepack.plot.tricontourf(floating, axes=axes[1], extend='both',
                                      levels=np.linspace(0, 1, 106))

    thin = icepack.interpolate(h - hLast, Q)
    thinC = icepack.plot.tricontourf(thin, levels=np.linspace(-8, 8, 161),
                                     axes=axes[2], extend='both',
                                     cmap=plt.get_cmap('bwr_r'))
    plotData['figM'].suptitle(f'Simulation Year: {t:0.1f}')
    if plotData['figMColorbar']:
        titles = ['melt (m/yr)', 'betaScale', 'dH/dt (m/yr)']
        for ax, cont, title in zip(axes, [meltC, floatC, thinC], titles):
            myColorBar(plotData['figM'], ax, cont)
            ax.set_title(title)
        plotData['figMColorbar'] = False
        plotData['figM'].tight_layout(rect=[0, 0, 1, 0.95])
    if plotData['plotResult']:
        plotData['figM'].canvas.draw()
    # plt.pause(0.01)


def combineA(A, theta, floating, grounded, forwardParams):
    '''
    '''
    if forwardParams['GLTaper'] > 0:
        print(f'GLTaper {forwardParams["GLTaper"]}')
        groundedSmooth = mf.firedrakeSmooth(grounded,
                                            alpha=forwardParams['GLTaper'])
        floatingSmooth = mf.firedrakeSmooth(floating,
                                            alpha=forwardParams['GLTaper'])
        AForward = A * groundedSmooth + floatingSmooth * firedrake.exp(theta)
    else:
        AForward = A * grounded + floating * firedrake.exp(theta)
    return AForward


def savePlots(plotData, forwardParams):
    ''' Save plots
    '''
    fNames = {'figM': 'mapFig', 'figP': 'profileFig', 'figTS': 'timeSeriesFig'}
    baseDir = forwardParams["forwardResultDir"]
    baseName = forwardParams["forwardResult"]
    for figKey in fNames:
        if plotData[figKey] is not None:
            plotData[figKey].savefig(
                f'{baseDir}/{fNames[figKey]}.{baseName}.png', dpi=200)
            if not plotData['plotResult']:
                plt.close(plotData[figKey])


def main():
    # declare globals - fix later
    # global mesh, floatingG, groundedG
    forwardParams, inversionParams = parsePigForwardArgs()
    #
    # Read mesh and setup function spaces
    mesh, Q, V, meshOpts = mf.setupMesh(forwardParams['mesh'],
                                        degree=forwardParams['degree'])
    opts = {}
    opts['dirichlet_ids'] = meshOpts['dirichlet_ids']  # Opts from mesh
    #
    beta0, theta0, A, s0, h0, zb, floating0, grounded0, uInv, uObs = \
        mf.getInversionData(forwardParams['inversionResult'], Q, V)
    AForward = combineA(A, theta0, floating0, grounded0, forwardParams)
    SMB = readSMB(forwardParams['SMB'], Q)
    meltModel, meltParams = setupMelt(forwardParams)
    # Setup ice stream model
    frictionLaw = setupFriction(forwardParams)
    #
    forwardModel = icepack.models.IceStream(friction=frictionLaw)
    forwardSolver = icepack.solvers.FlowSolver(forwardModel, **opts)
    # initial solve

    u0 = forwardSolver.diagnostic_solve(velocity=uObs, thickness=h0,
                                        surface=s0, fluidity=AForward,
                                        beta=beta0, grounded=grounded0,
                                        floating=floating0,
                                        uThresh=forwardParams['uThresh'])
    # copy original state
    h, hLast, s, u, zF, grounded, floating = \
        initialState(h0, s0, u0, zb, grounded0, floating0, Q)
    # meltParams['meltMask'] =  icepack.interpolate(h0 > 400, Q)
    summaryData = initSummary(grounded0, floating0, h0, u0, meltModel,
                              meltParams, Q, mesh)
    # Sanity/consistency check
    velocityError(u0, uInv, summaryData['area'], 'Difference with inversion')
    velocityError(u0, uObs, summaryData['area'], 'Difference with Obs')
    # setup plots
    plotData = {'plotResult': forwardParams['plotResult'],
                'mapPlotLimits': forwardParams['mapPlotLimits']}
    setupTimesSeriesPlots(plotData, forwardParams)
    setupProfilePlots(plotData, forwardParams, h)
    setupMapPlots(plotData, first=True)
    profilePlots(-1e-12, plotData, uObs, s0, h0, zb, zF, None, Q, first=True)
    #
    chk = setupOutputs(forwardParams, inversionParams)
    #
    beginTime = datetime.now()
    deltaT = forwardParams['deltaT']
    tBetaScale = 0  # FIXE  TEMPORARY
    betaScale = grounded * 1
    beta = beta0
    for t in np.arange(0, forwardParams['nYears'] + deltaT, deltaT):
        #
        melt = meltModel(h, floating, meltParams, Q, u)
        a = icepack.interpolate(SMB + melt, Q)
        if forwardParams['prognosticOnly']:
            u = uObs
        h = forwardSolver.prognostic_solve(forwardParams['deltaT'],
                                           thickness=h, velocity=u,
                                           accumulation=a,
                                           thickness_inflow=h0)
        h = icepack.interpolate(firedrake.max_value(10, h), Q)
        # Compute surface and masks
        s = computeSurface(h, zb, Q)
        floating, grounded = mf.flotationMask(s, zF, Q)
        # Scale beta near gl
        if t > tBetaScale:
            betaScale = mf.reduceNearGLBeta(s, s0, zF, grounded, Q,
                                            forwardParams['GLThresh'])
            # print(betaScale.dat.data_ro.max())
            beta = beta0 * betaScale
        else:
            beta = beta0
        u = forwardSolver.diagnostic_solve(velocity=u, thickness=h, surface=s,
                                           fluidity=AForward, beta=beta,
                                           uThresh=forwardParams['uThresh'],
                                           floating=floating,
                                           grounded=grounded)
        print('.', end='', flush=True)
        myDiv = firedrake.div(u * h)
        fDiv = firedrake.assemble(myDiv * floating * firedrake.dx)
        gDiv = firedrake.assemble(myDiv * grounded * firedrake.dx)
        Af = firedrake.assemble(a * floating * firedrake.dx)
        Ag = firedrake.assemble(a * grounded * firedrake.dx)
        print(gDiv * 1e-9, (Ag - gDiv )*(0.917)*1e-9  )
        #
        if computeSummaryData(summaryData, h, hLast, s, u, melt, grounded,
                              floating, t, Q, mesh, beginTime):
            if plotData['plotResult']:  # Only do final plot (below)
                mapPlots(plotData, t, melt, betaScale, h, hLast, Q)
            hLast = h.copy(deepcopy=True)  # Save for next summary calc
            # For now ouput fields at same interval as summary data
            outputTimeStep(t, chk,  h=h, s=s, u=u, grounded=grounded,
                           floating=floating)
        #
        timeSeriesPlots(t, plotData, summaryData)
        profilePlots(t, plotData, u, s, h, zb, zF, melt, Q)

    # End Simulation
    mapPlots(plotData, t, melt, betaScale, h, hLast, Q)
    saveSummaryData(forwardParams, summaryData)
    savePlots(plotData, forwardParams)
    if plotData['plotResult']:
        print('Show Plot')
        plt.show()


main()
