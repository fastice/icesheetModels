#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
from utilities import myerror, mywarning
import modelfunc as mf
import firedrake
import icepack
from icepack.constants import ice_density as rhoI
from datetime import datetime
# import icepack.inverse
import matplotlib.pyplot as plt
from matplotlib import cm
import icepack.plot
import icepack.models
# from firedrake import PETSc
import numpy as np
import yaml
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable
import warnings

rhoW = rhoI * 1028./917.  # This ensures rhoW based on 1028
waterToIce = 1000./917.
iceToWater = 917./1000.
waterToIce = 1000./917.
floatingG, groundedG, mesh = None, None, None

# These numbers derived from repeat of the Joughin et al 2019 experiments
# with data sets used in the current experiment. They have not been evaluted
# on ice streams other than PIG.
GLThreshDefaults = {'schoof': 41, 'weertman': 122}

# ----- Parse command and input file arguments -----


def parsePigForwardArgs():
    ''' Handle command line args'''
    defaults = {'geometry': 'PigGeometry.yaml',
                'degree': 1,
                'plotResult': False,
                'noOverwrite': False,
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
                'tBetaScale': 0.0,
                'restart': False,
                'calvingMask': None,
                'meltModel': 'piecewiseWithDepth',
                'profileFile': '/Volumes/UsersIan/ModelExperiments/'
                'PigForward/llprof/piglong.xy',
                'mapPlotLimits': {'xmin': -1.66e6, 'xmax': -1.51e6,
                                  'ymin': -3.50e5, 'ymax': -2.30e5},
                'meltRegionsFile': None,
                'meltRegions': None,
                'forwardSolverTolerance': 1e-10
                }
    parser = argparse.ArgumentParser(
        description='\n\n\033[1mRun a forward simulation initialized by an '
        'inversion \033[0m\n\n')
    parser.add_argument('--geometry', type=str, default=None,
                        help=f'Yaml file with geometry file info '
                        f'[{defaults["geometry"]}] ')
    parser.add_argument('--SMB', type=str, default=None,
                        help=f'Geotiff with SMB data '
                        f'[{defaults["SMB"]}] ')
    parser.add_argument('--degree', type=int, default=None,
                        choices=[1, 2], help='Degree for mesh ')
    parser.add_argument('--nYears', type=float, default=None,
                        help=f'Simulation length (yrs) [{defaults["nYears"]}]')
    parser.add_argument('--GLThresh', type=float, default=None,
                        help='Threshhold for GL weakening '
                        f'[{GLThreshDefaults}]')
    parser.add_argument('--forwardSolverTolerance', type=float, default=None,
                        help='Diagnostic solver tolerance '
                        f'[{defaults["forwardSolverTolerance"]}]')
    parser.add_argument('--meltParams', type=str, default=None,
                        help='Name of melt params from meltParams.yaml file '
                        f'[{defaults["meltParams"]}]')
    parser.add_argument('--restart', action='store_true', default=False,
                        help=f'Restart simulation[{defaults["restart"]}]')
    parser.add_argument('--meltParamsFile', type=str, default=None,
                        help='Yaml file with melt params'
                        f'[{defaults["meltParamsFile"]}]')
    parser.add_argument('--meltRegionsFile', type=str, default=None,
                        help='File specifying melt regions [None]')
    parser.add_argument('--meltAnomaly', type=rangeLimitedFloatType,
                        default=None, help='Amplitude of melt anomaly as '
                        f'fraction of mean [{defaults["meltAnomaly"]}]')
    parser.add_argument('--meltTrend', type=float, nargs=2,
                        default=None, help='Melt trend slope intercept '
                        f' [{defaults["meltTrend"]}]')
    parser.add_argument('--meltPeriod', type=rangeLimitedFloatType,
                        default=None, help='Period of sinusoidal melt anomaly'
                        f' in years[{defaults["meltPeriod"]}]')
    parser.add_argument('--tBetaScale', type=rangeLimitedFloatType,
                        default=None, help='Period before beta scale turns on'
                        f' in years[{defaults["tBetaScale"]}]')
    parser.add_argument('--calvingMask', type=str, default=None,
                        help='Tiff with area to remove from shelf '
                        '(set to thin ice)'
                        f'[{defaults["calvingMask"]}]')
    parser.add_argument('--deltaT', type=float, default=None,
                        help=f'Time step (yrs) [{defaults["deltaT"]}]')
    parser.add_argument('--plotResult', action='store_true',
                        default=defaults["plotResult"],
                        help=f'Display results [{defaults["plotResult"]}]')
    parser.add_argument('--noOverwrite', action='store_true',
                        default=defaults["noOverwrite"],
                        help='Do not overwrite exsting complete run'
                        f' [{defaults["noOverwrite"]}]')
    parser.add_argument('--params', type=str, default=None,
                        help=f'Input parameter file (.yaml)'
                        f'[{defaults["params"]}]')
    parser.add_argument('inversionResult', type=str, nargs=1,
                        help='Base name(.degX.yaml/.h5) for inversion result')
    parser.add_argument('forwardResult', type=str, nargs=1,
                        help='Base name forward output')
    #
    forwardParams, inversionParams = parseForwardParams(parser, defaults)
    print('\n\n**** FORWARD MODEL PARAMS ****')
    for key in forwardParams:
        print(f'{key}: {forwardParams[key]}')
    print('**** END MODEL PARAMS ****\n')
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
    for key in ['friction', 'degree', 'mesh', 'uThresh', 'GLTaper', 
                'solverTolerance']:
        try:
            forwardParams[key] = inversionParams[key]
        except Exception:
            myerror(f'parseForwardParams: parameter- {key} - missing from'
                    ' inversion result')
    if forwardParams['GLThresh'] is None:
        forwardParams['GLThresh'] = GLThreshDefaults[forwardParams['friction']]
    # for param in ['uThresh']:
    #    forwardParams[param] = firedrake.Constant(forwardParams[param])
    #
    summaryFile = f'{forwardParams["forwardResultDir"]}/' \
                  f'{forwardParams["forwardResult"]}.summary.yaml.tmp'
    #
    if not os.path.exists(summaryFile):
        mywarning(f'Cannot reinit with tmp summary {summaryFile}\n'
                  'Starting from scratch')
        forwardParams['restart'] = False
    #
    return forwardParams, inversionParams

# ----- Plot Stuff -----


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


def setupMapPlots(plotData, first=False):
    ''' Setup plot for map view of melt, grounded/floating, thinning.
    Call for each plot to clear memory
    '''
    # First call just set to None
    if first:

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


def mapPlots(plotData, t, melt, floating, h, hLast, deltaT, Q):
    ''' Plot melt, floating/grounded, thinning in map view
    '''
    setupMapPlots(plotData)
    axes = plotData['axesM'].flatten()
    meltC = icepack.plot.tricontourf(icepack.interpolate(melt, Q),
                                     levels=np.linspace(-150, 0, 151),
                                     extend='both', axes=axes[0])
    floatC = icepack.plot.tricontourf(floating, axes=axes[1], extend='both',
                                      levels=np.linspace(0, 1, 106))

    thin = icepack.interpolate((h - hLast) / deltaT, Q)
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


def timeSeriesPlots(year, plotData, summaryDataDict):
    ''' Plot time series data
    '''
    summaryData = summaryDataDict['All']
    if year % summaryData['dTsum'] > .0001:
        return
    t = np.array(summaryData['year'][0:-1])
    # Plot floating and grounded area
    axesTS = plotData['axesTS']
    dShelfArea = (np.array(summaryData['fArea'][0:-1]) -
                  summaryData['fArea'][0]) / 1e6
    axesTS[0].plot(t, dShelfArea, 'b.', label='dA floating')
    # Compute and plot volume changes
    # axesTS[2].plot(t, summaryData['Umax-U0'][-1], 'ro', label='Max speedup')
    if len(summaryData['deltaVF']) > 1:
        axesTS[1].plot(t, np.array(summaryData['deltaVF'][0:-2])/1e9, 'bo',
                       label='dV floating')
        axesTS[1].plot(t, np.array(summaryData['deltaVG'][0:-2])/1e9, 'ko',
                       label='dV grounded')
        axesTS[1].plot(t, np.array(summaryData['meltTot'][0:-1])/1e9, 'ro',
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
        axesTS[2].plot(t, np.array(summaryData['DVF'][0:-2])/1e9, 'cd',
                       label='Floating loss')
        axesTS[2].plot(t, np.array(summaryData['DVG'][0:-2])/1e9, 'md',
                       label='Grounded loss')
    if plotData['plotResult']:
        plt.draw()
        plt.pause(1.)
        plotData['figTS'].canvas.draw()


def setupProfilePlots(plotData, forwardParams, h, skip=False):
    ''' Setup axes for times series plots
    '''
    # If profile doesn't exist, return
    if not os.path.exists(forwardParams['profileFile']) or skip:
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


def myColorBar(fig, ax, contours, label=''):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(contours, cax=cax, label=label)


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


# ---- Compute Model Stats ----


def volumeChange(grounded, floating, h, u, a, mesh,
                 mask=firedrake.Constant(1)):
    ''' Compute volume change on grounded and floating ice. Summing deltaVG
    should automatically provide VAF since the loss is always grounded.
    '''
    fluxDiv = firedrake.div(u * h)
    # flux divergence
    # print(firedrake.assemble(floating * notCalved * firedrake.dx))
    # print(firedrake.assemble( floating * firedrake.dx))
    fluxDivFloating = firedrake.assemble(fluxDiv * mask * floating *
                                         firedrake.dx)
    fluxDivGrounded = firedrake.assemble(fluxDiv * mask * grounded *
                                         firedrake.dx)
    # net accumulation
    Af = firedrake.assemble(a * mask * floating * firedrake.dx)
    Ag = firedrake.assemble(a * mask * grounded * firedrake.dx)
    deltaVG = -fluxDivGrounded + Ag
    deltaVF = -fluxDivFloating + Af
    # deltaVG = firedrake.assemble(grounded * (h - hLast) * firedrake.dx(mesh))
    # deltaVF = firedrake.assemble(floating * (h - hLast) * firedrake.dx(mesh))
    # print(deltaVG/1e9, deltaVG1/1e9, deltaVF/1e9,deltaVF1/1e9,Ag/1e9, Af/1e9)
    return deltaVF, deltaVG


def computeSummaryData(SD, h, s, u, a, melt, SMB, grounded, floating,
                       year, Q, mesh, deltaT, beginTime,
                       mask=firedrake.Constant(1),
                       printStatus=False):
    ''' Compute summary results and sort in summaryData
        Save all as float for yaml output
    '''
    iceToWater = 917./1000.
    deltaVF, deltaVG = \
        volumeChange(grounded, floating, h, u, a, mesh, mask=mask)
    SD['deltaVF'][-1] += deltaVF * deltaT * iceToWater
    SD['deltaVG'][-1] += deltaVG * deltaT * iceToWater
    SD['DVF'][-1] += deltaVF * deltaT * iceToWater
    SD['DVG'][-1] += deltaVG * deltaT * iceToWater
    if printStatus:
        print(f"++{year:0.3f} {SD['deltaVF'][-1]/1e9:0.3f} "
              f"{SD['deltaVG'][-1]/1e9:0.3f} {SD['DVF'][-1]/1e9:0.3f} "
              f"{SD['DVG'][-1]/1e9:0.3f}")
        print(f'year {year} runtime {datetime.now() - beginTime}')
    #
    # If not with half a time step of year return
    # Need to update this for different update interval (~=1)
    dTint = abs(year - round(year, 0))  # difference from int year
    # if before first year or not with half time step of int year return
    if year < (1. - deltaT * 0.5) or not (dTint < deltaT * 0.5):
        return False
    #
    gArea = firedrake.assemble(mask * grounded * firedrake.dx(mesh))
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
    meltTot = firedrake.assemble(icepack.interpolate(mask * floating * melt,
                                                     Q) *
                                 firedrake.dx(mesh))
    SD['meltTot'].append(float(meltTot))
    SMBfloating = firedrake.assemble(icepack.interpolate(mask * floating * SMB,
                                                         Q) *
                                     firedrake.dx(mesh))
    SD['SMBfloating'].append(float(SMBfloating))
    SMBgrounded = firedrake.assemble(icepack.interpolate(mask * grounded * SMB,
                                                         Q) *
                                     firedrake.dx(mesh))
    SD['SMBgrounded'].append(float(SMBgrounded))
    return True


def updateSummaryData(summaries, h, s, u, a, melt, SMB, grounded, floating,
                      year, Q, mesh, deltaT, beginTime, regionMasks):
    '''
    Compute stats for basin as a whole and seperately for individual basins.
    '''
    for regionKey in regionMasks:
        printStatus = False
        if regionKey == 'All':
            printStatus = True
        returnStatus = computeSummaryData(summaries[regionKey], h, s, u, a,
                                          melt, SMB, grounded, floating, year,
                                          Q, mesh, deltaT, beginTime,
                                          mask=regionMasks[regionKey],
                                          printStatus=printStatus)
    return returnStatus

#
# ---- Restart Code ----


# def reinitSummary(forwardParams):
#     ''' Init with last temporary summary file '''
#     summaryFile = f'{forwardParams["forwardResultDir"]}/' \
#                   f'{forwardParams["forwardResult"]}.summary.yaml.tmp'
#     if not os.path.exists(summaryFile):
#         mywarning(f'Cannot reinit with tmp summary {summaryFile}\n'
#                   'Starting from scratch')
#         return None
#     with open(summaryFile, 'r') as fp:
#         mp = yaml.load(fp, Loader=yaml.FullLoader)
#     return mp['summaryFile']

def reinitSummary(forwardParams, region='All'):
    ''' Init with last temporary summary file '''
    if region == 'All':
        suffix = '.summary.yaml.tmp'
    else:
        suffix = f'.{region}.summary.yaml.tmp'
    #
    summaryFile = f'{forwardParams["forwardResultDir"]}/' \
        f'{forwardParams["forwardResult"]}{suffix}'
    #
    if not os.path.exists(summaryFile):
        mywarning(f'Cannot reinit with tmp summary {summaryFile}\n'
                  'Starting from scratch')
        return None
    with open(summaryFile, 'r') as fp:
        mp = yaml.load(fp, Loader=yaml.FullLoader)
    return mp['summaryFile']


def doRestart(startIdx, mesh, forwardParams, zb, deltaT, chk, Q, V):
    ''' Read in last state to restart simulation and regenerate summary data
    '''
    t = startIdx
    # chk.set_timestep(t, idx=index[-1])
    myVars = {}
    for varName in ['h', 's', 'floating', 'grounded', 'u']:
        myVars[varName] = chk.load_function(mesh, name=varName, idx=startIdx)
    #
    zF = mf.flotationHeight(zb, Q)
    hLast = myVars['h'].copy(deepcopy=True)
    print(f'restarting at {t}')
    return t + deltaT, myVars['h'], hLast, myVars['s'], myVars['u'], zF, \
        myVars['grounded'], myVars['floating']

# ------ Model Initialization/Setup stuff --------


def initSummary(grounded0, floating0, h0, u0, meltModel, meltParams, SMB, Q,
                mesh, forwardParams, regionMasks, restart=False):
    ''' Compute initial areas and summary data
        if restart, try reload restart data. If not go with clean slate, which
        should be a legacy case (from when intermediates steps were not saved)
    '''
    summaries = {'All': None}
    if forwardParams['meltRegions'] is not None:
        for key in forwardParams['meltRegions']:
            summaries[key] = None
    #
    for key in summaries:
        print(f'initializing {key}')
        if forwardParams['restart']:
            summaryData = reinitSummary(forwardParams, region=key)
            if summaryData is not None:
                summaries[key] = summaryData
                continue
            # No summary.tmp from a prior run so reset restart
            forwardParams['restart'] = False
        #
        mask = regionMasks[key]
        # start from scratch
        area = firedrake.assemble(mask * firedrake.Constant(1) *
                                  firedrake.dx(mesh))
        gArea0 = firedrake.assemble(mask * grounded0 * firedrake.dx(mesh))
        fArea0 = area - gArea0
        #
        melt = meltModel(h0, floating0, meltParams, Q, u0,
                         meltRegions=forwardParams['meltRegions'])
        #
        meltTot = firedrake.assemble(icepack.interpolate(mask * floating0 *
                                                         melt, Q) *
                                     firedrake.dx(mesh))
        SMBfloating = firedrake.assemble(icepack.interpolate(mask * floating0 *
                                                             SMB, Q) *
                                         firedrake.dx(mesh))
        SMBgrounded = firedrake.assemble(icepack.interpolate(mask * grounded0 *
                                                             SMB, Q) *
                                         firedrake.dx(mesh))
        summaryData = {'year': [0], 'DVG': [0, 0], 'DVF': [0, 0],
                       'deltaVF': [0, 0], 'deltaVG': [0, 0],
                       'gArea': [float(gArea0)], 'fArea': [float(fArea0)],
                       'meltTot': [float(meltTot)], 'area': float(area),
                       'SMBgrounded': [float(SMBgrounded)],
                       'SMBfloating': [float(SMBfloating)],
                       'dTsum': 1.}  # dTsum fixed
        summaries[key] = summaryData
    return summaries


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


def setupTaperedMasks(myParams, grounded, floating):
    ''' Smooth or copy floating and grounded masks for tapering near gl
    '''
    # global floatingSmooth, groundedSmooth
    if myParams['GLTaper'] < 1:
        floatingSmooth = floating.copy(deepcopy=True)
        groundedSmooth = grounded.copy(deepcopy=True)
    else:
        groundedSmooth = mf.firedrakeSmooth(grounded,
                                            alpha=myParams['GLTaper'])
        floatingSmooth = mf.firedrakeSmooth(floating,
                                            alpha=myParams['GLTaper'])
    return groundedSmooth, floatingSmooth


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

# ---- Melt Scaling -----


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

# ---- Assorted Model Functions


def taperedViscosity(velocity, thickness, fluidity, theta,
                     groundedSmooth, floatingSmooth):
    ''' This version uses feathered grouning to floating transition
    '''
    # Combine tapered A on grouned and theta for smooth
    A = groundedSmooth * fluidity + floatingSmooth * firedrake.exp(theta)
    viscosity = \
        icepack.models.viscosity.viscosity_depth_averaged(velocity=velocity,
                                                          thickness=thickness,
                                                          fluidity=A)
    return viscosity


def computeSurface(h, zb, Q):
    '''Hack of icepack version to uses different rhoI/rhoW
    '''
    s = firedrake.max_value(h + zb, h * (1 - rhoI/rhoW))
    return icepack.interpolate(s, Q)


def checkThickness(h, thresh, Q, t, h0):
    ''' Do not let ice get too thin
    '''
    #
    h = icepack.interpolate(firedrake.max_value(thresh, h), Q)
    return h


# ---- Output Results ----

def setupOutputs(forwardParams, inversionParams, meltParams, check=True):
    ''' Make output dir and dump forward and inversionParams
    '''
    if not os.path.exists(forwardParams['forwardResultDir']):
        os.mkdir(forwardParams['forwardResultDir'])
    inputsFile = f'{forwardParams["forwardResultDir"]}/' \
                 f'{forwardParams["forwardResult"]}.inputs.yaml'
    chkFile = f'{forwardParams["forwardResultDir"]}/' \
              f'{forwardParams["forwardResult"]}.history'
    summaryFile = f'{forwardParams["forwardResultDir"]}/' \
                  f'{forwardParams["forwardResult"]}.summary.yaml'
    # Abort if noOverwrite and all files exist
    noOverwrite = forwardParams['noOverwrite']
    for myFile in [inputsFile, f'{chkFile}.h5', summaryFile]:
        noOverwrite = noOverwrite and os.path.exists(myFile)
        print(myFile, noOverwrite)
    if noOverwrite:
        myerror(f'\nProducts exist for: {inputsFile}\nand noOverwrite set\n')
    # Continue with new or overwrite
    print(f'Writing inputs to: {inputsFile}')
    #
    with open(inputsFile, 'w') as fpYaml:
        myDicts = {'forwardParams': forwardParams,
                   'inversionParams': inversionParams,
                   'meltParams': meltParams}
        yaml.dump(myDicts, fpYaml)
    # open check point file
    if check:
        if forwardParams['restart']:
            mode = 'a'
        else:
            mode = 'w'
        return firedrake.CheckpointFile(f'{chkFile}.h5', mode), chkFile
    return None, None


def saveSummaryData(forwardParams, summaryDataDict, tmpFile=False):
    ''' Write summary data to yaml file
    '''
    # loop over regions
    for region in summaryDataDict:
        # setup name for region
        if region == 'All':
            suffix = '.summary.yaml'
        else:
            suffix = f'.{region}.summary.yaml'
    #
        summaryFile = f'{forwardParams["forwardResultDir"]}/' \
            f'{forwardParams["forwardResult"]}{suffix}'
        #
        summaryData = summaryDataDict[region]
        if tmpFile:
            summaryFile += '.tmp'
        # Trim last values, which were used for summation of next step
        if not tmpFile:  # Only trim final value
            for key in ['deltaVF', 'deltaVG', 'DVG', 'DVF']:
                summaryData[key] = summaryData[key][0:-1]
                if os.path.exists(f'{summaryFile}.tmp'):  # Final, remove tmp
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


def outputTimeStep(idx, mesh, chk, **kwargs):
    ''' Ouput variables at a time step)
    '''
    # chk.set_timestep(t)
    for k in kwargs:
        chk.save_function(kwargs[k], name=k, idx=idx)


def printExtremes(**kwargs):
    ''' Print min/max of firedrake functions to flag bad inputs'''
    print('Min/Max of input values')
    print(''.join(['-']*40))
    for arg in kwargs:
        print(arg, kwargs[arg].dat.data_ro.min(),
              kwargs[arg].dat.data_ro.max())
    print(''.join(['-']*40))


def getMeshFromCheckPoint(checkFile):
    '''
    If a new checkpoint file, read the mesh from there.
    '''
    if '.h5' not in checkFile:
        checkFile = f'{checkFile}.h5'
    print(checkFile)

    with firedrake.CheckpointFile(checkFile, 'r') as chk:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=DeprecationWarning)
            try:
                mesh = chk.load_mesh()
                return mesh
            except Exception:
                return None


def readChkVariables(myVars, mesh, chk):
    '''
    Read a list of check vars.
    '''
    result = []
    for myVar in myVars:
        result.append(chk.load_function(mesh, myVar))
    return result


def saveChkVariables(myVars, mesh, chk):
    result = []
    for myVar in myVars:
        result.append(chk.save_function(myVars[myVar], name=myVar))
    return result


def loadMeltRegions(forwardParams, mesh, Q):
    '''
        Load the melt regions if they exist
    '''
    if forwardParams['meltRegionsFile'] is None:
        return
    meltRegionsFiles = mf.readModelParams(forwardParams['meltRegionsFile'])
    meltRegions = {}
    print('*** Melt Scaling ***')
    for key in meltRegionsFiles['meltRegionFiles']:
        print(f'Reading {meltRegionsFiles["meltRegionFiles"][key]}')
        meltRegions[key] = mf.getModelVarFromTiff(
            meltRegionsFiles['meltRegionFiles'][key], Q)
    forwardParams['meltRegions'] = meltRegions


def getRegionMasks(forwardParams):
    '''
    Create masks based on melt masks for computing stats for individual basins
    '''
    masks = {'All': firedrake.Constant(1)}
    if forwardParams['meltRegions'] is not None:
        for key in forwardParams['meltRegions']:
            masks[key] = forwardParams['meltRegions'][key]
    return masks

# ----- Main ----


def main():
    ''' Main for foward model simulation '''
    forwardParams, inversionParams = parsePigForwardArgs()
    #
    meltModel, meltParams = setupMelt(forwardParams)
    #
    chk, chkFile = setupOutputs(forwardParams, inversionParams, meltParams)
    if not forwardParams['restart']:
        meshI = getMeshFromCheckPoint(forwardParams['inversionResult'])
    else:
        meshI = chk.load_mesh()
        print(meshI)
    #
    mesh, Q, V, meshOpts = \
        mf.setupMesh(forwardParams['mesh'], degree=forwardParams['degree'],
                     meshOversample=inversionParams['meshOversample'],
                     newMesh=meshI)
    #
    # Setup ice stream model
    frictionLaw = setupFriction(forwardParams)
    #
    forwardModel = icepack.models.IceStream(friction=frictionLaw,
                                            viscosity=taperedViscosity)
    opts = {'dirichlet_ids': meshOpts['dirichlet_ids'],
            'diagnostic_solver_parameters':
                {'max_iterations': 150,
                 'tolerance': forwardParams['forwardSolverTolerance']}}
    forwardSolver = icepack.solvers.FlowSolver(forwardModel, **opts)
    uThresh = firedrake.Constant(forwardParams['uThresh'])
    #
    initVarNames = ['beta0', 'theta0', 'A0', 's0', 'h0', 'zb', 'floating0',
                    'grounded0', 'uInv', 'uObs', 'groundedSmooth',
                    'floatingSmooth', 'SMB', 'u0']
    # Load melt regions if they exist
    loadMeltRegions(forwardParams, mesh, Q)
    regionMasks = getRegionMasks(forwardParams)
    #
    print(f'Restart status {forwardParams["restart"]}')
    #
    # Read mesh and setup function spaces
    if not forwardParams['restart']:
        #
        chk.save_mesh(mesh)
        beta0, theta0, A0, s0, h0, zb, floating0, grounded0, uInv, uObs = \
            mf.getInversionData(forwardParams['inversionResult'], Q, V,
                                mesh=mesh)
        # Compute masks for combining A0 with theta0
        groundedSmooth, floatingSmooth = \
            setupTaperedMasks(inversionParams, grounded0, floating0)
        SMB = readSMB(forwardParams['SMB'], Q)
        # initial solve
        u0 = forwardSolver.diagnostic_solve(velocity=uObs, thickness=h0,
                                            surface=s0, fluidity=A0,
                                            beta=beta0, theta=theta0,
                                            grounded=grounded0,
                                            floating=floating0,
                                            groundedSmooth=groundedSmooth,
                                            floatingSmooth=floatingSmooth,
                                            uThresh=uThresh)
        initVar = dict(zip(initVarNames,
                           [beta0, theta0, A0, s0, h0, zb, floating0,
                            grounded0, uInv, uObs, groundedSmooth,
                            floatingSmooth, SMB, u0]))
        saveChkVariables(initVar, mesh, chk)
    else:
        beta0, theta0, A0, s0, h0, zb, floating0, \
            grounded0, uInv, uObs, groundedSmooth, \
            floatingSmooth, SMB, u0 = readChkVariables(initVarNames, mesh, chk)
    #
    # Get fresh or reloaded summary data - reset restart if not data
    summaryData = initSummary(grounded0, floating0, h0, u0, meltModel,
                              meltParams,  SMB, Q, mesh, forwardParams,
                              regionMasks)
    #
    deltaT = forwardParams['deltaT']
    # copy original state
    if not forwardParams['restart']:
        startYear = 0
        h, hLast, s, u, zF, grounded, floating = \
            initialState(h0, s0, u0, zb, grounded0, floating0, Q)
    else:  # load state to restart
        startIdx = int(summaryData['All']['year'][-1])
        if startIdx < 1:
            myerror(f"Could not get start year for restart: {startIdx}")
        startYear, h, hLast, s, u, zF, grounded, floating = \
            doRestart(startIdx, mesh, forwardParams, zb, deltaT, chk, Q, V)
    #
    # Sanity/consistency check
    print(f'area {summaryData["All"]["area"]}')
    mf.velocityError(u0, uInv, summaryData["All"]['area'],
                     'Difference with uInv')
    mf.velocityError(u0, uObs, summaryData["All"]['area'],
                     'Difference with uObs')
    mf.velocityError(uInv, uObs, summaryData["All"]['area'],
                     'Difference with uInv/uObs')
    # setup plots
    plotData = {'plotResult': forwardParams['plotResult'],
                'mapPlotLimits': forwardParams['mapPlotLimits'],
                'figM': None, 'axesM': None}
    setupTimesSeriesPlots(plotData, forwardParams)
    setupProfilePlots(plotData, forwardParams, h, skip=True)
    # profilePlots(-1e-12, plotData, uObs, s0, h0, zb, zF, None, Q, first=True)
    #
    beginTime = datetime.now()
    betaScale = grounded * 1
    beta = beta0
    print('Loop')
    if startYear > forwardParams['nYears']:
        myerror(f'startYear ({startYear}) is greater than nYears '
                f'({forwardParams["nYears"]}). Restart '
                f'{forwardParams["nYears"]} so sim may be done')
    times = np.arange(startYear, forwardParams['nYears'] + deltaT, deltaT)
    for t in np.around(times, decimals=6):
        #
        hLast = h.copy(deepcopy=True)  # Save for next summary calc
        trend = meltTrend(t, forwardParams)
        melt = meltModel(h, floating, meltParams, Q, u,
                         trend=trend,
                         meltRegions=forwardParams['meltRegions']) * \
            meltAnomaly(t, forwardParams)
        # exit()
        a = icepack.interpolate((SMB + melt) * waterToIce, Q)
        #
        h = forwardSolver.prognostic_solve(forwardParams['deltaT'],
                                           thickness=h, velocity=u,
                                           accumulation=a,
                                           thickness_inflow=h0)
        # Don't allow to go too thin.
        h = checkThickness(h, 30, Q, t, h0)
        # Compute surface and masks
        s = computeSurface(h, zb, Q)
        floating, grounded = mf.flotationMask(s, zF, Q)
        # Scale beta near gl
        if t > forwardParams['tBetaScale']:
            betaScale = mf.reduceNearGLBeta(s, s0, zF, grounded, Q,
                                            forwardParams['GLThresh'])
            beta = icepack.interpolate(beta0 * betaScale, Q)
        else:
            beta = beta0
        uThresh = firedrake.Constant(forwardParams['uThresh'])
        u = forwardSolver.diagnostic_solve(velocity=u, thickness=h, surface=s,
                                           fluidity=A0, beta=beta,
                                           theta=theta0,
                                           uThresh=uThresh,
                                           floating=floating,
                                           grounded=grounded,
                                           groundedSmooth=groundedSmooth,
                                           floatingSmooth=floatingSmooth)
        print('.', end='', flush=True)
        #
        # Compute summary data and plot if indicated
        if updateSummaryData(summaryData, h, s, u, a, melt, SMB, grounded,
                             floating, t, Q, mesh, deltaT, beginTime,
                             regionMasks):
            if plotData['plotResult']:  # Only do final plot (below)
                mapPlots(plotData, t, melt, betaScale, h, hLast, deltaT, Q)
            # For now ouput fields at same interval as summary data
            outputTimeStep(int(np.round(t)), mesh, chk,  h=h, s=s, u=u,
                           grounded=grounded, floating=floating)
            # write after h5 so last year recorded
            saveSummaryData(forwardParams, summaryData, tmpFile=True)
        #

        # profilePlots(t, plotData, u, s, h, zb, zF, melt, Q)
    #
    # End Simulation so save results
    timeSeriesPlots(t, plotData, summaryData)
    mapPlots(plotData, t, melt, betaScale, h, hLast, deltaT, Q)
    saveSummaryData(forwardParams, summaryData, tmpFile=False)
    savePlots(plotData, forwardParams)
    # Show final result if requested
    if plotData['plotResult']:
        print('Show Plot')
        plt.show()


main()
