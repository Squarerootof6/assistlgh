import math
import astropy.units as u
import astropy.constants as c
import numpy as np
from numba import jit,prange
from assistlgh.arepo_processing.read_parameter import AllPara
def gmc_starformation_criteria(i, SphP, All):
    """_summary_

    Args:
        i (_int_): order of sph paricle list
        SphP (_h5py_): Sph particle list
        All (_AllPara_): parameter list

    Returns:
        _int_: starformation criteria flag in RIGEL
    """
    HYDROGEN_MASSFRAC=1
    BOLTZMANN=1.38065e-16
    PROTONMASS=1.67262178e-24
    GAMMA =(5. / 3.) #adiabatic index of simulated gas
    GAMMA_MINUS1=(GAMMA - 1.)
    GRAVITY = 6.6738e-8
    meanweight = 4.0 / (1+3*HYDROGEN_MASSFRAC); #for neutral gas
    temperature_prefac = All.UnitEnergy_in_cgs / All.UnitMass_in_g * GAMMA_MINUS1 * meanweight * PROTONMASS / BOLTZMANN;
    crit_flag = np.ones(len(i))
    # force star formation if gas density reaches GMCDensMax
    mask = SphP['Density'][i] > All.GMCDensMax
    crit_flag[np.logical_and(mask,crit_flag==1)] = 2
    # temperature threshold
    uthermal = SphP['InternalEnergy'][i]*SphP['Masses'][i]
    mask = temperature_prefac * uthermal > All.GMCTempThres
    crit_flag[np.logical_and(mask,crit_flag==1)] = 0
    # density threshold
    mask = SphP['Density'][i] < All.GMCDensThres
    crit_flag[np.logical_and(mask,crit_flag==1)] = 0

    # Jeans threshold (Necessary condition)
    # Yunwei suggests 10
    '''temperature = (SphP['InternalEnergy'][i]*All.UnitEnergy_in_cgs*2/3*PROTONMASS/BOLTZMANN*(1/2*SphP['HI_Fraction'][i]+1/2))
    ndensity = (SphP['Density'][i]*All.UnitMass_in_g/All.UnitLength_in_cm**3/PROTONMASS)
    pressure = (ndensity*BOLTZMANN*temperature)/All.UnitMass_in_g/All.UnitVelocity_in_cm_per_s**2*All.UnitLength_in_cm**3'''
    pressure = (GAMMA-1)*SphP['Density'][i]*All.UnitMass_in_g/All.UnitLength_in_cm**3*SphP['InternalEnergy'][i]*All.UnitEnergy_in_cgs
    cs2 = GAMMA * pressure / SphP['Density'][i]
    if All.GravityConstantInternal == 0:
        All.G = GRAVITY / pow(All.UnitLength_in_cm, 3) * All.UnitMass_in_g * pow(All.UnitTime_in_s, 2);
    else:
        All.G = All.GravityConstantInternal
    mJeans = math.pi / 6.0 * np.sqrt((3.0 / 32.0 * cs2 * math.pi / All.G)**3 / SphP['Density'][i])
    mask = mJeans > All.GMCNumJeansMassNecessary * SphP['Masses'][i]
    crit_flag[np.logical_and(mask,crit_flag==1)] = 0
    # force star formation if Jeans mass is not resolved (Sufficient condition)
    # Yunwei suggests 0.12 from Grudic 21, assume f_J = 0.5
    mask = mJeans < All.GMCNumJeansMassSufficient * SphP['Masses'][i]
    crit_flag[np.logical_and(mask,crit_flag==1)] = 2
    # self-gravitational e.g. Hopkins+13
    sigma_turb2 = SphP['VelocityDivergence'][i] * SphP['VelocityDivergence'][i] + SphP['VelocityCurl'][i] * SphP['VelocityCurl'][i]
    # cs2 = GAMMA * SphP[i].Pressure / SphP[i].Density;
    # alpha_selfgrav = All.GMCSelfGravFactor * (sigma_turb2+cs2) / All.GravityConstantInternal / SphP[i].Density;
    # print(f"GMC_SFR: cell_id={i}, div={SphP[i].VelocityDivergence}, curl={SphP[i].VelocityCurl}")
    alpha_selfgrav = All.GMCSelfGravFactor * sigma_turb2 / All.G / SphP['Density'][i]
    mask = alpha_selfgrav > 1
    crit_flag[np.logical_and(mask,crit_flag==1)] = 0
        
    # divergence flow
    mask = SphP['VelocityDivergence'][i]>0
    crit_flag[np.logical_and(mask,crit_flag==1)] = 0
    # potential peak?? TO DO
    return crit_flag

@jit(fastmath=True,nopython=True)
def gmc_starformation_criteria_single(pressure,density,InternalEnergy,Masses,VelocityDivergence,VelocityCurl,UnitEnergy_in_cgs,UnitMass_in_g,UnitTime_in_s,UnitLength_in_cm,GMCDensMax,GMCTempThres,GMCDensThres,GMCNumJeansMassNecessary,GMCNumJeansMassSufficient,GravityConstantInternal,GMCSelfGravFactor,verbose=False):
    """_summary_
    Args:
        i (_int_): order of sph paricle list
        SphP (_h5py_): Sph particle list
        All (_AllPara_): parameter list
    Returns:
        _int_: starformation criteria flag in RIGEL
    """
    HYDROGEN_MASSFRAC=1;BOLTZMANN=1.38065e-16;PROTONMASS=1.67262178e-24;GAMMA =(5. / 3.);GAMMA_MINUS1=(GAMMA - 1.);GRAVITY = 6.6738e-8
    meanweight = 4.0 / (1+3*HYDROGEN_MASSFRAC); 
    temperature_prefac = UnitEnergy_in_cgs / UnitMass_in_g * GAMMA_MINUS1 * meanweight * PROTONMASS / BOLTZMANN;
    crit_flag = 1
    # force star formation if gas density reaches GMCDensMax
    if density > GMCDensMax:
        if verbose:
            print('density > GMCDensMax')
        crit_flag = 2
        return crit_flag
    # temperature threshold
    uthermal = InternalEnergy
    if temperature_prefac * uthermal > GMCTempThres:
        if verbose:
            print('temperature_prefac * uthermal > GMCTempThres')
        crit_flag = 0
        return crit_flag
    # density threshold
    if density < GMCDensThres:
        if verbose:
            print('density < GMCDensThres')
        crit_flag = 0
        return crit_flag
    # Jeans threshold (Necessary condition)
    # Yunwei suggests 10
    #pressure = (GAMMA-1)*density*UnitMass_in_g/UnitLength_in_cm**3*InternalEnergy*UnitEnergy_in_cgs
    cs2 = GAMMA * pressure / density
    if GravityConstantInternal == 0:
        G = GRAVITY / pow(UnitLength_in_cm, 3) * UnitMass_in_g * pow(UnitTime_in_s, 2);
    else:
        G = GravityConstantInternal
    mJeans = math.pi / 6.0 * np.sqrt((3.0 / 32.0 * cs2 * math.pi / G)**3 / density)
    if mJeans > GMCNumJeansMassNecessary * Masses:
        if verbose:
            print('mJeans > GMCNumJeansMassNecessary * Masses',mJeans,GMCNumJeansMassNecessary,Masses)
        crit_flag = 0
        return crit_flag
    # force star formation if Jeans mass is not resolved (Sufficient condition)
    # Yunwei suggests 0.12 from Grudic 21, assume f_J = 0.5
    if mJeans < GMCNumJeansMassSufficient * Masses:
        if verbose:
            print('mJeans < GMCNumJeansMassSufficient * Masses')
        crit_flag = 2
        return crit_flag
    # self-gravitational e.g. Hopkins+13
    sigma_turb2 = VelocityDivergence * VelocityDivergence + VelocityCurl * VelocityCurl
    alpha_selfgrav = GMCSelfGravFactor * sigma_turb2 / G / density
    if alpha_selfgrav > 1:
        if verbose:
            print('alpha_selfgrav > 1')
        crit_flag = 0
        return crit_flag
    # divergence flow
    if VelocityDivergence > 0:
        if verbose:
            print('VelocityDivergence>0')
        crit_flag = 0
        return crit_flag
    return crit_flag
    
@jit(fastmath=True,nopython=True)
def main_loop(Number,pressure,density,InternalEnergy,Masses,VelocityDivergence,VelocityCurl,UnitEnergy_in_cgs,UnitMass_in_g,UnitTime_in_s,UnitLength_in_cm,GMCDensMax,GMCTempThres,GMCDensThres,GMCNumJeansMassNecessary,GMCNumJeansMassSufficient,GravityConstantInternal,GMCSelfGravFactor):
    mask_sf = np.zeros(Number)
    for i in range(Number):
        mask_sf[i] = gmc_starformation_criteria_single(pressure[i],density[i],InternalEnergy[i],Masses[i],VelocityDivergence[i],VelocityCurl[i],UnitEnergy_in_cgs,UnitMass_in_g,UnitTime_in_s,UnitLength_in_cm,GMCDensMax,GMCTempThres,GMCDensThres,GMCNumJeansMassNecessary,GMCNumJeansMassSufficient,GravityConstantInternal,GMCSelfGravFactor)
    return mask_sf
def decide_sf(snap,outputdir):
    para = AllPara('./'+outputdir+'/param.txt')
    UnitEnergy_in_cgs = para.UnitEnergy_in_cgs
    UnitMass_in_g = para.UnitMass_in_g
    UnitTime_in_s = para.UnitTime_in_s
    UnitLength_in_cm = para.UnitLength_in_cm
    GMCDensMax = para.GMCDensMax
    GMCTempThres = para.GMCTempThres
    GMCDensThres = para.GMCDensThres
    GMCNumJeansMassNecessary = para.GMCNumJeansMassNecessary
    GMCNumJeansMassSufficient = para.GMCNumJeansMassSufficient
    GravityConstantInternal = para.GravityConstantInternal
    GMCSelfGravFactor = para.GMCSelfGravFactor
    Masses = snap['0_Masses'].value
    pressure = snap['0_Pressure'].value
    density = snap['0_Density'].value
    InternalEnergy = snap['0_InternalEnergy'].value
    VelocityDivergence = snap['0_VelocityDivergence'].value
    VelocityCurl = snap['0_VelocityCurl'].value
    mask_sf = main_loop(len(Masses),pressure,density,InternalEnergy,Masses,VelocityDivergence,VelocityCurl,UnitEnergy_in_cgs,UnitMass_in_g,UnitTime_in_s,UnitLength_in_cm,GMCDensMax,GMCTempThres,GMCDensThres,GMCNumJeansMassNecessary,GMCNumJeansMassSufficient,GravityConstantInternal,GMCSelfGravFactor)
    mask_sf = mask_sf.astype(bool)
    return mask_sf