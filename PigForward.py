#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
from utilities import myerror, mywarning
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
waterToIce = 1000./917.
iceToWater = 917./1000.
floatingG, groundedG, mesh = None, None, None

# These numbers derived from repeat of the Joughin et al 2019 experiments
# with data sets used in the current experiment. They have not been evaluted
# on ice streams other than PIG.
GLThreshDefaults = {'schoof': 41, 'weertman': 122}


def parsePigForwardArgs():
    ''' Handle command line args'''
    defaults = {'geometry': 'PigGeometry.yaml',
                'degree': 1,
                'plotResult': False,
                'params': None,
                'inversionResult': None,
                'nYears': 10.,
                'GLThresh': None,  # Optimal for PIG schoof
                'SMB': '/home/ian/ModelRuns/Thwaites/BrookesMap/'
                'OLS_Trend_plus_Resid_9b9.tif',
                'deltaT': 0.05,
                'meltParamsFile': 'meltParams.yaml',
                'meltParams': 'linear',
                'meltAnomaly': 0.0,
                'meltTrend': None,
                'meltPeriod': None,
                'restart': False,
                'calvingMask': None,
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
                        f'[{GLThreshDefaults}]')
    parser.add_argument('--meltParams', type=str, default=None,
                        help='Name of melt params from meltParams.yaml file '
                        f'[{defaults["meltParams"]}]')
    parser.add_argument('--restart', action='store_true', default=False,
                        help=f'Restart simulation[{defaults["restart"]}]')
    parser.add_argument('--meltParamsFile', type=str, default=None,
                        help='Yaml file with melt params'
                        f'[{defaults["meltParamsFile"]}]')
    parser.add_argument('--meltAnomaly', type=rangeLimitedFloatType,
                        default=None, help='Amplitude of melt anomaly as '
                        f'fraction of mean [{defaults["meltAnomaly"]}]')
    parser.add_argument('--meltTrend', type=float, nargs=2,
                        default=None, help='Melt trend slope intercept '
                        f' [{defaults["meltTrend"]}]')
    parser.add_argument('--meltPeriod', type=rangeLimitedFloatType,
                        default=None, help='Period of sinusoidal melt anomaly'
                        f' in years[{defaults["meltPeriod"]}]')
    parser.add_argument('--calvingMask', type=str, default=None,
                        help='Tiff with area to remove from shelf '
                        '(set to thin ice)'
                        f'[{defaults["calvingMask"]}]')
    parser.add_argument('--deltaT', type=float, default=None,
                        help=f'Time step (yrs) [{defaults["deltaT"]}]')
    parser.add_argument('--plotResult', action='store_true',
                        default=defaults["plotResult"],
                        help=f'Display results [{defaults["plotResult"]}]')
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


def rangeLimitedFloatType(arg):
    try:
        f = float(arg)
    except ValueError:
        raise argparse.ArgumentTypeError('Must be a float')
    if f <= 0. or f > 1000.:
        raise argparse.ArgumentTypeError('Must be > 0 and < 1000')
    return f


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
    if forwardParams['GLThresh'] is None:
        forwardParams['GLThresh'] = GLThreshDefaults[forwardParams['friction']]
    return forwardParams, inversionParams


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
    yLabels = ['dA (km^2)', 'dV (GT/yr)', 'Tot. Vol. Change (GT)']
    # axesTS = np.append(axesTS, axesTS[1].twinx())
    for ax, yLabel in zip(axesTS, yLabels):
        ax.set_xlim([0, forwardParams['nYears']])
        ax.set_xlabel('t (years)')
        ax.set_ylabel(yLabel)
    plt.tight_layout()
    plotData['figTS'], plotData['axesTS'] = figTS, axesTS


def volumeChange(grounded, floating, notCalved, h, u, a, mesh):
    ''' Compute volume change on grounded and floating ice. Summing deltaVG
    should automatically provide VAF since the loss is always grounded.
    '''
    fluxDiv = firedrake.div(u * h)
    # flux divergence
    fluxDivFloating = firedrake.assemble(fluxDiv * floating * notCalved *
                                         firedrake.dx)
    fluxDivGrounded = firedrake.assemble(fluxDiv * grounded * firedrake.dx)
    # net accumulation
    Af = firedrake.assemble(a * floating * notCalved * firedrake.dx)
    Ag = firedrake.assemble(a * grounded * firedrake.dx)
    deltaVG = -fluxDivGrounded + Ag
    deltaVF = -fluxDivFloating + Af
    # deltaVG = firedrake.assemble(grounded * (h - hLast) * firedrake.dx(mesh))
    # deltaVF = firedrake.assemble(floating * (h - hLast) * firedrake.dx(mesh))
    # print(deltaVG/1e9, deltaVG1/1e9, deltaVF/1e9,deltaVF1/1e9,Ag/1e9, Af/1e9)
    return deltaVF, deltaVG


def reinitSummary(forwardParams):
    ''' Init with last temporary summary file '''
    summaryFile = f'{forwardParams["forwardResultDir"]}/' \
                  f'{forwardParams["forwardResult"]}.summary.yaml.tmp'
    if not os.path.exists(summaryFile):
        mywarning('Cannot reinit with tmp summary {summaryFile}')
        return None
    with open(summaryFile, 'r') as fp:
        mp = yaml.load(fp, Loader=yaml.FullLoader)
    # summaryData = mp['summaryFile']
    # Temp file will already have these values appended
    # for key in ['deltaVF', 'deltaVG']:
    #    summaryData[key].append(0)
    # for key in ['DVF', 'DVG']:
    #    summaryData[key].append(summaryData[key][-1])
    return mp['summaryFile']


def initSummary(grounded0, floating0, h0, u0, meltModel, meltParams, SMB, Q,
                mesh, forwardParams, restart=False):
    ''' Compute initial areas and summary data
        if restart, try reload restart data. If not go with clean slate, which
        should be a legacy case (from when intermediates steps were not saved)
    '''
    if restart:
        summaryData = reinitSummary(forwardParams)
        if summaryData is not None:
            return summaryData
    # start from scratch
    area = firedrake.assemble(firedrake.Constant(1) * firedrake.dx(mesh))
    gArea0 = firedrake.assemble(grounded0 * firedrake.dx(mesh))
    fArea0 = area - gArea0
    melt = meltModel(h0, floating0, meltParams, Q, u0)
    meltTot = firedrake.assemble(icepack.interpolate(floating0 * melt, Q) *
                                 firedrake.dx(mesh))
    SMBfloating = firedrake.assemble(icepack.interpolate(floating0 * SMB, Q) *
                                     firedrake.dx(mesh))
    SMBgrounded = firedrake.assemble(icepack.interpolate(grounded0 * SMB, Q) *
                                     firedrake.dx(mesh))
    summaryData = {'year': [0], 'DVG': [0, 0], 'DVF': [0, 0],
                   'deltaVF': [0, 0], 'deltaVG': [0, 0],
                   'gArea': [float(gArea0)], 'fArea': [float(fArea0)],
                   'meltTot': [float(meltTot)], 'area': float(area),
                   'SMBgrounded': [float(SMBgrounded)],
                   'SMBfloating': [float(SMBfloating)],
                   'dTsum': 1.}
    return summaryData


def doRestart(forwardParams, zb, deltaT, chk, Q, V):
    ''' Read in last state to restart simulation and regenerate summary data
    '''
    tSteps, index = chk.get_timesteps()
    t = tSteps[-1]
    chk.set_timestep(t, idx=index[-1])
    myVars = {}
    for varName in ['h', 's', 'floating', 'grounded']:
        myVar = firedrake.Function(Q, name=varName)
        chk.load(myVar, name=varName)
        myVars[varName] = myVar
    #
    myVar = firedrake.Function(V, name='u')
    chk.load(myVar, name='u')
    myVars['u'] = myVar
    zF = mf.flotationHeight(zb, Q)
    hLast = myVars['h'].copy(deepcopy=True)
    return t + deltaT, myVars['h'], hLast, myVars['s'], myVars['u'], zF, \
        myVars['grounded'], myVars['floating']


def computeSummaryData(SD, h, hLast, s, u, a, melt, SMB, grounded, floating,
                       notCalved, year, Q, mesh, deltaT, beginTime):
    ''' Compute summary results and sort in summaryData
        Save all as float for yaml output
    '''
    deltaVF, deltaVG = \
        volumeChange(grounded, floating, notCalved, h, u, a, mesh)
    SD['deltaVF'][-1] += deltaVF * deltaT * iceToWater
    SD['deltaVG'][-1] += deltaVG * deltaT * iceToWater
    SD['DVF'][-1] += deltaVF * deltaT * iceToWater
    SD['DVG'][-1] += deltaVG * deltaT * iceToWater
    print(f"++{year:0.3f} {SD['deltaVF'][-1]/1e9:0.3f} "
          f"{SD['deltaVG'][-1]/1e9:0.3f} {SD['DVF'][-1]/1e9:0.3f} "
          f"{SD['DVG'][-1]/1e9:0.3f}")
    #
    if year % SD['dTsum'] > .0001 * year or year < SD['dTsum']:
        return False
    print(f'year {year} runtime {datetime.now() - beginTime}')
    #
    gArea = firedrake.assemble(grounded * firedrake.dx(mesh))
    SD['year'].append(float(year))
    SD['gArea'].append(gArea)
    SD['fArea'].append(SD['area'] - gArea)
    # append a new zero value to start incrementing
    SD['deltaVF'].append(0)
    SD['deltaVG'].append(0)
    # append current value to start incrementing
    SD['DVF'].append(SD['DVF'][-1])
    SD['DVG'].append(SD['DVG'][-1])
    #
    meltTot = firedrake.assemble(icepack.interpolate(floating * melt, Q) *
                                 firedrake.dx(mesh))
    SD['meltTot'].append(float(meltTot))
    SMBfloating = firedrake.assemble(icepack.interpolate(floating * SMB, Q) *
                                     firedrake.dx(mesh))
    SD['SMBfloating'].append(float(SMBfloating))
    SMBgrounded = firedrake.assemble(icepack.interpolate(grounded * SMB, Q) *
                                     firedrake.dx(mesh))
    SD['SMBgrounded'].append(float(SMBgrounded))
    #
    print(f'{year}: Initial melt {SD["meltTot"][0] / 1e9:.2f} current melt '
          f' {meltTot / 1e9:.2f}')
    return True


def saveSummaryData(forwardParams, summaryData, tmp=False):
    ''' Write summary data to yaml file
    '''
    summaryFile = f'{forwardParams["forwardResultDir"]}/' \
                  f'{forwardParams["forwardResult"]}.summary.yaml'
    if tmp:
        summaryFile += '.tmp'
    # Trim last values, which were used for summation of next step
    if not tmp:  # Only trim final value
        for key in ['deltaVF', 'deltaVG', 'DVG', 'DVF']:
            summaryData[key] = summaryData[key][0:-1]
            if os.path.exists(f'{summaryFile}.tmp'):  # Final, so remove tmp
                os.remove(f'{summaryFile}.tmp')
    # Convert numpy to list
    for s in summaryData:
        if isinstance(summaryData[s], np.ndarray):
            summaryData[s] = summaryData[s].tolist()
        # convert list elements out of np
        if isinstance(summaryData[s], list):
            tmp = []
            for x in summaryData[s]:
                if isinstance(x, (np.generic)):
                    x = x.item()
                tmp.append(x)
            summaryData[s] = tmp
    # Now write result to yaml
    with open(summaryFile, 'w') as fpYaml:
        yaml.dump({'summaryFile': summaryData}, fpYaml)


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
    if len(summaryData['deltaVF']) > 1:
        axesTS[1].plot(t, summaryData['deltaVF'][-2]/1e9, 'bo',
                       label='dV floating')
        axesTS[1].plot(t, summaryData['deltaVG'][-2]/1e9, 'ko',
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
    if len(summaryData['deltaVF']) > 1:
        axesTS[2].plot(t, summaryData['DVF'][-2]/1e9, 'cd',
                       label='Floating loss')
        axesTS[2].plot(t, summaryData['DVG'][-2]/1e9, 'md',
                       label='Grounded loss')
    if plotData['plotResult']:
        plt.draw()
        plt.pause(1.)
        plotData['figTS'].canvas.draw()


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


def setupOutputs(forwardParams, inversionParams, meltParams, check=True):
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
                   'inversionParams': inversionParams,
                   'meltParams': meltParams}
        yaml.dump(myDicts, fpYaml)
    # open check point file
    chkFile = f'{forwardParams["forwardResultDir"]}/' \
              f'{forwardParams["forwardResult"]}.history'
    if check:
        if forwardParams['restart']:
            mode = firedrake.FILE_UPDATE
        else:
            mode = firedrake.FILE_CREATE
        return firedrake.DumbCheckpoint(chkFile, mode=mode)
    return None


def outputTimeStep(t, chk,  **kwargs):
    ''' Ouput variables at a time step)
    '''
    chk.set_timestep(t)
    for k in kwargs:
        chk.store(kwargs[k], name=k)


def readSMB(SMBfile, Q):
    ''' Read SMB file an limit values to +/- 6 to avoid no data values

    Returns water equivalent values.
    '''
    if not os.path.exists:
        myerror(f'readSMB: SMB file  ({SMBfile}) does not exist')
    SMB = mf.getModelVarFromTiff(SMBfile, Q)
    # avoid any unreasonably large value
    SMB = icepack.interpolate(
        firedrake.max_value(firedrake.min_value(SMB, 6), -6), Q)
    return SMB


def readCalved(calvingMask, floating, Q):
    ''' Read a file with mask indicating a section of the shelf to remove.
    '''
    if calvingMask is None:
        calved = icepack.interpolate(floating < -10, Q)  # Force all 0s
        notCalved = icepack.interpolate(calved > -10, Q)
        return calved, notCalved
    # Read file if path specified
    if not os.path.exists:
        myerror(f'readCalved: Calving mask ({calvingMask}) does not exist')
    calved = mf.getModelVarFromTiff(calvingMask, Q)
    # avoid any unreasonably large value
    calved = icepack.interpolate(
        firedrake.max_value(firedrake.min_value(calved, 1), 0), Q)
    notCalved = icepack.interpolate(calved < 0.5, Q)
    return calved, notCalved


def myColorBar(fig, ax, contours, label=''):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(contours, cax=cax, label=label)


def mapPlots(plotData, t, melt, floating, h, hLast, Q):
    ''' Plot melt, floating/grounded, thinning in map view
    '''
    setupMapPlots(plotData)
    axes = plotData['axesM'].flatten()
    meltC = icepack.plot.tricontourf(icepack.interpolate(melt, Q),
                                     levels=np.linspace(-150, 0, 151),
                                     extend='both', axes=axes[0])
    floatC = icepack.plot.tricontourf(floating, axes=axes[1], extend='both',
                                      levels=np.linspace(0, 1, 106))

    thin = icepack.interpolate(h - hLast, Q)
    thinC = icepack.plot.tricontourf(thin, levels=np.linspace(-8, 8, 161),
                                     axes=axes[2], extend='both',
                                     cmap=plt.get_cmap('bwr_r'))
    plotData['figM'].suptitle(f'Simulation Year: {t:0.1f}')
    if plotData['figMColorbar']:
        titles = ['melt (m/yr)', 'floating', 'dH/dt (m/yr)']
        for ax, cont, title in zip(axes, [meltC, floatC, thinC], titles):
            myColorBar(plotData['figM'], ax, cont)
            ax.set_title(title)
        plotData['figMColorbar'] = False
        plotData['figM'].tight_layout(rect=[0, 0, 1, 0.95])
    if plotData['plotResult']:
        plotData['figM'].canvas.draw()
    # plt.pause(0.01)


def combineA(A, theta, floating, grounded, forwardParams, Q):
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
        AForward = icepack.interpolate(A * grounded +
                                       floating * firedrake.exp(theta), Q)
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


def meltAnomaly(y, forwardParams):
    '''
    Compute melt anomaly as 1 + meltAnomaly * sin(t/meltPeriod)
    Parameters
    ----------
    y: float
        year of simumlation
    forwardParams : dict
        forward params.
    Returns
    -------
    melt scale factor.
    '''
    meltScale = 1.
    if forwardParams['meltPeriod'] is not None:
        meltScale += forwardParams['meltAnomaly'] * \
            np.sin(y/forwardParams['meltPeriod'] * 2.0 * np.pi)
    return firedrake.Constant(meltScale)


def meltTrend(y, forwardParams):
    '''
    Compute melt trend as intercept + y * slope
    Parameters
    ----------
    y: float
        year of simumlation
    forwardParams : dict
        forward params.
    Returns
    -------
    melt scale factor.
    '''
    meltTrend = forwardParams['meltTrend']
    if meltTrend is None:
        return 0.
    return (meltTrend[0] + meltTrend[1] * y) * 1e9


def checkThickness(h, calved, notCalved, thresh, Q):
    ''' Set region marked as calved to nominal thin ice value given by thresh
    '''
    h = calved * thresh + notCalved * h
    h = icepack.interpolate(firedrake.max_value(thresh, h), Q)
    return h


def main():
    # declare globals - fix later
    # global mesh, floatingG, groundedG
    waterToIce = 1./0.917
    forwardParams, inversionParams = parsePigForwardArgs()
    print(forwardParams)
    #
    # Read mesh and setup function spaces
    mesh, Q, V, meshOpts = \
        mf.setupMesh(forwardParams['mesh'], degree=forwardParams['degree'],
                     meshOversample=inversionParams['meshOversample'])
    #
    beta0, theta0, A, s0, h0, zb, floating0, grounded0, uInv, uObs = \
        mf.getInversionData(forwardParams['inversionResult'], Q, V)
    AForward = combineA(A, theta0, floating0, grounded0, forwardParams, Q)
    SMB = readSMB(forwardParams['SMB'], Q)
    calved, notCalved = readCalved(forwardParams['calvingMask'], floating0, Q)
    meltModel, meltParams = setupMelt(forwardParams)
    # Setup ice stream model
    frictionLaw = setupFriction(forwardParams)
    #
    forwardModel = icepack.models.IceStream(friction=frictionLaw)
    opts = {}
    opts['dirichlet_ids'] = meshOpts['dirichlet_ids']  # Opts from mesh
    opts['diagnostic_solver_parameters'] = {'max_iterations': 150}
    forwardSolver = icepack.solvers.FlowSolver(forwardModel, **opts)
    # initial solve
    u0 = forwardSolver.diagnostic_solve(velocity=uObs, thickness=h0,
                                        surface=s0, fluidity=AForward,
                                        beta=beta0, grounded=grounded0,
                                        floating=floating0,
                                        uThresh=forwardParams['uThresh'])
    #
    chk = setupOutputs(forwardParams, inversionParams, meltParams)
    deltaT = forwardParams['deltaT']
    # copy original state
    if not forwardParams['restart']:
        startYear = 0
        h, hLast, s, u, zF, grounded, floating = \
            initialState(h0, s0, u0, zb, grounded0, floating0, Q)
    else:  # load state to restart
        startYear, h, hLast, s, u, zF, grounded, floating = \
            doRestart(forwardParams, zb, deltaT, chk, Q, V)
        print(type(s))
        print(type(u))
        print(f'restarting at {startYear}')
    # meltParams['meltMask'] =  icepack.interpolate(h0 > 400, Q)
    summaryData = initSummary(grounded0, floating0, h0, u0, meltModel,
                              meltParams,  SMB, Q, mesh, forwardParams,
                              restart=forwardParams['restart'])
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
    beginTime = datetime.now()

    tBetaScale = 0  # FIXE  TEMPORARY
    betaScale = grounded * 1
    beta = beta0
    print('Loop')
    if startYear > forwardParams['nYears']:
        myerror(f'startYear ({startYear}) is greater than nYears '
                f'({forwardParams["nYears"]}). Restart '
                f'{forwardParams["nYears"]} so sim may be done')
    for t in np.arange(startYear, forwardParams['nYears'] + deltaT, deltaT):
        #
        lasttime = datetime.now()
        trend = meltTrend(t, forwardParams)
        melt = meltModel(h, floating, meltParams, Q, u, trend=trend) * \
            meltAnomaly(t, forwardParams)
        a = icepack.interpolate((SMB + melt) * waterToIce, Q)
        #
        newtime =datetime.now()
        print('Before prog', newtime-lasttime)
        lasttime = newtime

        h = forwardSolver.prognostic_solve(forwardParams['deltaT'],
                                           thickness=h, velocity=u,
                                           accumulation=a,
                                           thickness_inflow=h0)
        newtime =datetime.now()
        print('After prog', newtime-lasttime)
        lasttime = newtime

        # Don't allow to go too thin.
        h = checkThickness(h, calved, notCalved, 30, Q)
        # Compute surface and masks
        s = computeSurface(h, zb, Q)
        floating, grounded = mf.flotationMask(s, zF, Q)
        # Scale beta near gl
        if t > tBetaScale:
            betaScale = mf.reduceNearGLBeta(s, s0, zF, grounded, Q,
                                            forwardParams['GLThresh'])
            beta = icepack.interpolate(beta0 * betaScale, Q)
        else:
            beta = beta0

        newtime =datetime.now()
        print('before diag', newtime-lasttime)
        lasttime = newtime

        u = forwardSolver.diagnostic_solve(velocity=u, thickness=h, surface=s,
                                           fluidity=AForward, beta=beta,
                                           uThresh=forwardParams['uThresh'],
                                           floating=floating,
                                           grounded=grounded)
        newtime =datetime.now()
        print('After diag', newtime-lasttime)
        lasttime = newtime

        print('.', end='', flush=True)
        #
        if computeSummaryData(summaryData, h, hLast, s, u, a, melt, SMB,
                              grounded, floating, notCalved, t, Q, mesh,
                              deltaT, beginTime):
            if plotData['plotResult']:  # Only do final plot (below)
                mapPlots(plotData, t, melt, betaScale, h, hLast, Q)
            hLast = h.copy(deepcopy=True)  # Save for next summary calc
            # For now ouput fields at same interval as summary data
            outputTimeStep(t, chk,  h=h, s=s, u=u, grounded=grounded,
                           floating=floating)
            saveSummaryData(forwardParams, summaryData, tmp=True)
        #
        timeSeriesPlots(t, plotData, summaryData)
        profilePlots(t, plotData, u, s, h, zb, zF, melt, Q)

        newtime =datetime.now()
        print('End loop', newtime-lasttime)
        lasttime = newtime

    # End Simulation
    mapPlots(plotData, t, melt, betaScale, h, hLast, Q)
    saveSummaryData(forwardParams, summaryData)
    savePlots(plotData, forwardParams)
    if plotData['plotResult']:
        print('Show Plot')
        plt.show()


main()
