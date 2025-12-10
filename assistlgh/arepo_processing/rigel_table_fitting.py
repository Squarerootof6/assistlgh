import numpy as np
from scipy.optimize import brentq, newton, root_scalar
from scipy.interpolate import interpn, interp1d
from numba import njit,prange


def hardness(mass):
    if (mass<=40):
        return np.max([-0.1604+0.0093*mass, 0]);
    else:
        return np.min([0.1861+0.0007*mass, 1]);


def fromfit_4(mass, fit, mup, logy, corr):
    logm = np.log10(mass)
    logmup = np.log10(mup);
    y=0
    if mass>mup:
        for i in range(5):
            y += fit[i]*pow(logmup, i);
        y += (logm-logmup)*(fit[1]+2*fit[2]*logmup+3*fit[3]*logmup*logmup+4*fit[4]*logmup*logmup*logmup);
    else: 
         for i in range(5):
            y += fit[i]*pow(logm, i);

    if (corr==1):
        y = pow(10, y)
        y *= (1-hardness(mass));
        y = np.log10(y)
    if (corr==2):
        y = pow(10, y)
        y *= hardness(mass);
        y = np.log10(y)
    if logy>0:
        y = pow(10, y);
    return y;


def get_Z_indfit(metallicity, lZ, N_Z):
    for i in range(N_Z-1):
        if ((metallicity>=lZ[i]) and (metallicity<lZ[i+1])):
            return i;
        

ql0={}
ql0['N_Z'] = 6
ql0['lZ'] = np.array([1e-08, 1e-06, 1e-05, 0.0001, 0.01, 1.0])
ql0['mup'] = np.array([100,100,100,100,100,100,100])
ql0['qion'] = np.array([
    [46.46137570981074,0.5170813521553708,0.7980458845476294,-0.20001293338495338, 0.0],
    [46.46137570981074,0.5170813521553708,0.7980458845476294,-0.20001293338495338, 0.0],
    [45.42059040667134,2.700712335383665,-0.5300347502919636,0.057301896061572676, 0.0],
    [45.269082346617076,3.0666670430502267,-0.7162720521814226,0.089391565259882, 0.0],
    [45.18736137209817,3.3196846240547733,-0.8453761141197472,0.11198940085710157, 0.0],
    [45.1501073427151,3.593673871633107,-0.9739095691444815,0.1337856849577229, 0.0],
    [45.1841914991652,3.7649346618225934,-1.0450855792477083,0.14410831962830245, 0.0]
])

ql1={}
ql1['N_Z'] = 6
ql1['lZ'] = np.array([1e-08, 1e-06, 1e-05, 0.0001, 0.01, 1.0])
ql1['mup'] = np.array([100,100,100,100,100,100,100])
ql1['qion'] = np.array([
    [45.49959761035941,2.4128700886668,-0.17959424167711194,-0.02752103519559344, 0.0],
    [45.49959761035941,2.4128700886668,-0.17959424167711194,-0.02752103519559344, 0.0],
    [44.63425911392677,4.137941105064039,-1.192197832384118,0.16346765367229385, 0.0],
    [44.45086800985704,4.489495856250076,-1.3544675399009931,0.1892411663405467, 0.0],
    [44.287920931707866,4.809796952695418,-1.5055805570396352,0.2141221206446955, 0.0],
    [44.02214655602073,5.314930088000796,-1.7280992974633875,0.24907569739824917, 0.0],
    [43.746297744919005,5.818143510083072,-1.9412659403228654,0.28056869111217914, 0.0]
])

qlw={}
qlw['N_Z'] = 3
qlw['lZ'] = np.array([1e-08, 0.0001, 0.005])
qlw['mup'] = np.array([100,100,120,65])
qlw['qion'] = np.array([[44.791568579666404, 3.3897905892537823, -0.7466156809776126, 0.08427300524755489, 0.0], 
[44.791568579666404, 3.3897905892537823, -0.7466156809776126, 0.08427300524755489, 0.0],
[42.749092401201665,6.383004425534019,-2.222806887624244,0.32788391135378747, 0.0],
[36.19994400587021,19.19762726339443,-10.731382246611043,2.1555636902782807, 0.0]])

qhi={}
qhi['N_Z'] = 4
qhi['lZ'] = np.array([1e-8, 0.001, 0.02, 1])
qhi['mup'] = np.array([500, 500, 100, 100, 90])
qhi['qion'] = np.array([[43.50402000488536, 5.079483417442792, -1.157928059068298, 0.08609538811093954, 0.0],
[43.50402000488536, 5.079483417442792, -1.157928059068298, 0.08609538811093954, 0.0], 
[10.411647398234967, 83.07729930277493, -68.82805214236834, 25.726196429956868, -3.5979073218967055],
[27.429527763373063, 31.731604659134554, -15.694079269865117, 2.7150959710105296, 0.0],
[15.02741330958072, 62.91837414195692, -46.06678089761328, 15.927469836238958, -2.139219842335498]])




qhei={}
qhei['N_Z'] = 4
qhei['lZ'] = np.array([1e-08, 0.001, 0.02, 1])
qhei['mup'] = np.array([500,500,50,100,50])
qhei['qion'] = np.array([[42.01142307752654, 6.619996726966662, -1.5432531844293464, 0.09133565826436674, 0.0],
[42.01142307752654, 6.619996726966662, -1.5432531844293464, 0.09133565826436674, 0.0],
[-35.921049143315585, 181.88884307165338, -148.65081118047166, 54.4013642195234, -7.434995613420139],
[17.708316177372236, 45.7862856779324, -22.80714772341938, 3.9063425218161942, 0.0],
[-24.12740265442154, 119.97713398286368, -66.22448352943196, 12.311443282782571, 0.0]])

qheii={}
qheii['N_Z'] = 6
qheii['lZ'] = np.array([1e-08, 1e-06, 1e-05, 0.0001, 0.01, 1])
qheii['mup'] = np.array([100,100, 100, 100, 100, 100, 100])
qheii['qion'] = np.array([[26.50611069337394,29.07352127486989,-13.652937785114979,2.317705552679573, 0.0
],
[26.50611069337394,29.07352127486989,-13.652937785114979,2.317705552679573, 0.0
],
[28.712508893744747, 23.18721519103398, -9.556019601868615, 1.4504023134109312, 0.0 
],
[28.304973036367524, 22.963426477877224, -9.204968101348152, 1.3593382788672672, 0.0
],
[27.247277758174693,23.831995925067506,-9.478090487259445,1.386425977355938, 0.0
],
[24.357254542446427, 26.684605583986404, -10.598211806833403, 1.5381552201488347, 0.0
],
[20.485632381017602,30.767582589694012,-12.313924631665769,1.7903716018357156, 0.0
]])

mw_V={}
mw_V['N_Z'] = 5
mw_V['lZ'] = np.array([1e-06, 1e-05, 0.0001, 0.01, 1.0])
mw_V['mup'] = np.array([100,100,100,100,100])
mw_V['windrate']  = np.array([
[-19.662335779112972,15.201603518194537,-6.563165582113039,1.058381865080407, 0],
[-19.837540698040506,15.471257099701129,-6.472958627899068,1.015546373831702,0],
[-20.131176643072187,15.956004319476591,-6.534990490522251,1.0056633773510386,0],
[-20.767823788020806,16.935698096461827,-6.693800382974077,0.9965039119584684,0],
[-21.519745098459495, 17.97914687667549, -6.88374781560359, 0.990662876532161,0]
])

vw_V={}
vw_V['N_Z'] = 5
vw_V['lZ'] = np.array([1e-06, 1e-05, 0.0001, 0.01, 1.0])
vw_V['mup'] = np.array([100,100,100,100,100])
vw_V['windrate']  = np.array([
[2.1687595883214015,0.4850455402554444,-0.1391089043149817,0.009470193635398388,0],
[2.387602325351114,0.3951137507718151,-0.09058157404847247,0.0007687524336589249,0],
[2.5756640103362183,0.35826545969743057,-0.07150998970970389,-0.002518156519856758,0],
[2.9233740224091225,0.3322517829438451,-0.05741624741252023,-0.0052647967973936186,0],
[3.2530865595982585, 0.3322517829438432, -0.05741624741251887, -0.005264796797393941,0]
])



vw_S={}
vw_S['N_Z'] = 5
vw_S['lZ'] = np.array([1e-06, 1e-05, 0.0001, 0.01, 1.0])
vw_S['mup'] = np.array([100,100,100,100,100])
vw_S['windrate']  = np.array([
[3.371709302602629,0.26846785956863845,-0.023910446086444354,0.004808316398309251,0],
[3.4155634496386433,0.14199455288345558,0.04396976742381718,-0.00733728379350906,0],
[3.423625658283575,0.07510656168632494,0.0799655571223272,-0.013834825870429444,0],
[3.3898739529984967,0.027991786368639346,0.10508767261311361,-0.018638552503142742,0],
[3.3219179221729984, 0.027991786368650757, 0.10508767261310603, -0.01863855250314114,0]
])

mw_S={}
mw_S['N_Z'] = 5
mw_S['lZ'] = np.array([1e-06, 1e-05, 0.0001, 0.01, 1.0])
mw_S['mup'] = np.array([100,100,100,100,100])
mw_S['windrate']  = np.array([
[-19.070314400747137,6.790130939737788,-1.2560190435622114,0.061159029317066606,0],
[-18.407652315680977,6.827977051428192,-1.2736469166423905,0.0641207838709217,0],
[-17.789421447530763,6.958826938443637,-1.3519086411747616,0.07985808231994398,0],
[-16.46508068218042,7.050391665838392,-1.3984254325470955,0.08821791780963001,0],
[-15.065080682180433, 7.050391665838412, -1.398425432547108, 0.08821791780963256,0]
])

LT={}
LT['N_Z'] = 6
LT['lZ'] = np.array([1e-8, 0.0315, 0.315, 0.63, 1.575, 3.937])
LT['mup'] = np.array([500, 120, 120, 120, 120, 120])
LT['windrate']  = np.array([
[9.875, -3.759, 1.413, -0.186, 0],
[10.10369661, -3.78067428, 1.26441548, -0.1350663, 0],
[10.33606251, -4.19609554, 1.49306541, -0.17326104, 0],
[10.39142876, -4.28477573, 1.51191149, -0.16656313, 0],
[10.99849116, -5.71786943, 2.51517758, -0.38645937, 0],
[11.19124744, -6.50995816, 3.10807831, -0.51255932, 0]
])

def get_OPT(mass,metallicity):
    if (metallicity <ql0['lZ'][0]):
        return fromfit_4(mass, ql0['qion'][0], ql0['mup'][0], 1, 0);
    if (metallicity>=ql0['lZ'][ql0['N_Z']-1]):
        return fromfit_4(mass, ql0['qion'][ql0['N_Z']], ql0['mup'][ql0['N_Z']], 1, 0);
    i_Z = get_Z_indfit(metallicity, ql0['lZ'], ql0['N_Z']);
    dlogZ = np.log10(metallicity/ql0['lZ'][i_Z])/np.log10(ql0['lZ'][i_Z+1]/ql0['lZ'][i_Z]);
    return pow(10, (1-dlogZ)*fromfit_4(mass, ql0['qion'][i_Z+1], ql0['mup'][i_Z+1], 0, 0) + dlogZ*fromfit_4(mass, ql0['qion'][i_Z+2], ql0['mup'][i_Z+2], 0, 0));

def get_FUV(mass,metallicity):
    if (metallicity <ql1['lZ'][0]):
        return fromfit_4(mass, ql1['qion'][0], ql1['mup'][0], 1, 0);
    if (metallicity>=ql1['lZ'][ql1['N_Z']-1]):
        return fromfit_4(mass, ql1['qion'][ql1['N_Z']], ql1['mup'][ql1['N_Z']], 1, 0);
    i_Z = get_Z_indfit(metallicity, ql1['lZ'], ql1['N_Z']);
    dlogZ = np.log10(metallicity/ql1['lZ'][i_Z])/np.log10(ql1['lZ'][i_Z+1]/ql1['lZ'][i_Z]);
    return pow(10, (1-dlogZ)*fromfit_4(mass, ql1['qion'][i_Z+1], ql1['mup'][i_Z+1], 0, 0) + dlogZ*fromfit_4(mass, ql1['qion'][i_Z+2], ql1['mup'][i_Z+2], 0, 0));

def get_LW(mass,metallicity):
    if (metallicity <qlw['lZ'][0]):
        return fromfit_4(mass, qlw['qion'][0], qlw['mup'][0], 1, 0);
    if (metallicity>=qlw['lZ'][qlw['N_Z']-1]):
        return fromfit_4(mass, qlw['qion'][qlw['N_Z']], qlw['mup'][qlw['N_Z']], 1, 0);
    i_Z = get_Z_indfit(metallicity, qlw['lZ'], qlw['N_Z']);
    dlogZ = np.log10(metallicity/qlw['lZ'][i_Z])/np.log10(qlw['lZ'][i_Z+1]/qlw['lZ'][i_Z]);
    return pow(10, (1-dlogZ)*fromfit_4(mass, qlw['qion'][i_Z+1], qlw['mup'][i_Z+1], 0, 0) + dlogZ*fromfit_4(mass, qlw['qion'][i_Z+2], qlw['mup'][i_Z+2], 0, 0));

def get_UV0(mass,metallicity):
    i_Z, corr=0,0
    if (metallicity <qhi['lZ'][0]):
        return fromfit_4(mass, qhi['qion'][0], qhi['mup'][0], 1, 0);
    if (metallicity>=qhi['lZ'][qhi['N_Z']-1]):
        return fromfit_4(mass, qhi['qion'][qhi['N_Z']], qhi['mup'][qhi['N_Z']], 1, 0);
    i_Z = get_Z_indfit(metallicity, qhi['lZ'], qhi['N_Z']);
    dlogZ = np.log10(metallicity/qhi['lZ'][i_Z])/np.log10(qhi['lZ'][i_Z+1]/qhi['lZ'][i_Z]);
    return pow(10, (1-dlogZ)*fromfit_4(mass, qhi['qion'][i_Z+1], qhi['mup'][i_Z+1], 0, 0) + dlogZ*fromfit_4(mass, qhi['qion'][i_Z+2], qhi['mup'][i_Z+2], 0, 0));

def get_UV1(mass,metallicity):
    i_Z, corr=0,0
    if (metallicity <qhei['lZ'][0]):
        return fromfit_4(mass, qhei['qion'][0], qhei['mup'][0], 1, 0);
    if (metallicity>=qhei['lZ'][qhei['N_Z']-1]):
        return fromfit_4(mass, qhei['qion'][qhei['N_Z']], qhei['mup'][qhei['N_Z']], 1, 0);
    i_Z = get_Z_indfit(metallicity, qhei['lZ'], qhei['N_Z']);
    dlogZ = np.log10(metallicity/qhei['lZ'][i_Z])/np.log10(qhei['lZ'][i_Z+1]/qhei['lZ'][i_Z]);
    return pow(10, (1-dlogZ)*fromfit_4(mass, qhei['qion'][i_Z+1], qhei['mup'][i_Z+1], 0, 0) + dlogZ*fromfit_4(mass, qhei['qion'][i_Z+2], qhei['mup'][i_Z+2], 0, 0));

def get_UV2(mass, metallicity):
    i_Z, corr=0,0
    if (metallicity <qheii['lZ'][0]): 
        return fromfit_4(mass, qheii['qion'][0], qheii['mup'][0], 1, 0);
    if (metallicity>= qheii['lZ'][qheii['N_Z']-1]) :
        return fromfit_4(mass, qheii['qion'][qheii['N_Z']], qheii['mup'][qheii['N_Z']], 1, 0);
    i_Z = get_Z_indfit(metallicity, qheii['lZ'], qheii['N_Z']);
    dlogZ = np.log10(metallicity/qheii['lZ'][i_Z])/np.log10(qheii['lZ'][i_Z+1]/qheii['lZ'][i_Z]);
    return pow(10, (1-dlogZ)*fromfit_4(mass, qheii['qion'][i_Z+1], qheii['mup'][i_Z+1], 0, 0) + dlogZ*fromfit_4(mass, qheii['qion'][i_Z+2], qheii['mup'][i_Z+2], 0, 0));

def get_mdot(mass, metallicity):
    i_Z,corr=0,0
    if (metallicity <mw_V['lZ'][0]):
        return 0.0; 
    if (metallicity>=mw_V['lZ'][mw_V['N_Z']-1]):
        return fromfit_4(mass, mw_V['windrate'][mw_V['N_Z']-1], mw_V['mup'][mw_V['N_Z']-1], 1, 0); 
    i_Z = get_Z_indfit(metallicity, mw_V['lZ'], mw_V['N_Z']);
    dlogZ = np.log10(metallicity/mw_V['lZ'][i_Z])/np.log10(mw_V['lZ'][i_Z+1]/mw_V['lZ'][i_Z]);
    return pow(10, (1-dlogZ)*fromfit_4(mass, mw_V['windrate'][i_Z], mw_V['mup'][i_Z], 0, 0) + dlogZ*fromfit_4(mass, mw_V['windrate'][i_Z+1], mw_V['mup'][i_Z+1], 0, 0)); 

def get_velWind(mass, metallicity):
    i_Z,corr=0,0
    if (metallicity <vw_V['lZ'][0]):
        return 0.0; 
    if (metallicity>=vw_V['lZ'][vw_V['N_Z']-1]):
        return fromfit_4(mass, vw_V['windrate'][vw_V['N_Z']-1], vw_V['mup'][vw_V['N_Z']-1], 1, 0); 
    i_Z = get_Z_indfit(metallicity, vw_V['lZ'], vw_V['N_Z']);
    dlogZ = np.log10(metallicity/vw_V['lZ'][i_Z])/np.log10(vw_V['lZ'][i_Z+1]/vw_V['lZ'][i_Z]);
    return pow(10, (1-dlogZ)*fromfit_4(mass, vw_V['windrate'][i_Z], vw_V['mup'][i_Z], 0, 0) + dlogZ*fromfit_4(mass, vw_V['windrate'][i_Z+1], vw_V['mup'][i_Z+1], 0, 0));


def get_mdotS(mass, metallicity):
    i_Z,corr=0,0
    if (metallicity <mw_S['lZ'][0]):
        return 0.0; 
    if (metallicity>=mw_S['lZ'][mw_S['N_Z']-1]):
        return fromfit_4(mass, mw_S['windrate'][mw_S['N_Z']-1], mw_S['mup'][mw_S['N_Z']-1], 1, 0); 
    i_Z = get_Z_indfit(metallicity, mw_S['lZ'], mw_S['N_Z']);
    dlogZ = np.log10(metallicity/mw_S['lZ'][i_Z])/np.log10(mw_S['lZ'][i_Z+1]/mw_S['lZ'][i_Z]);
    return pow(10, (1-dlogZ)*fromfit_4(mass, mw_S['windrate'][i_Z], mw_S['mup'][i_Z], 0, 0) + dlogZ*fromfit_4(mass, mw_S['windrate'][i_Z+1], mw_S['mup'][i_Z+1], 0, 0)); 

def get_velWindS(mass, metallicity):
    i_Z,corr=0,0
    if (metallicity <vw_S['lZ'][0]):
        return 0.0; 
    if (metallicity>=vw_S['lZ'][vw_S['N_Z']-1]):
        return fromfit_4(mass, vw_S['windrate'][vw_S['N_Z']-1], vw_S['mup'][vw_S['N_Z']-1], 1, 0); 
    i_Z = get_Z_indfit(metallicity, vw_S['lZ'], vw_S['N_Z']);
    dlogZ = np.log10(metallicity/vw_S['lZ'][i_Z])/np.log10(vw_S['lZ'][i_Z+1]/vw_S['lZ'][i_Z]);
    return pow(10, (1-dlogZ)*fromfit_4(mass, vw_S['windrate'][i_Z], vw_S['mup'][i_Z], 0, 0) + dlogZ*fromfit_4(mass, vw_S['windrate'][i_Z+1], vw_S['mup'][i_Z+1], 0, 0)); 

def get_LT(mass, metallicity):
    i_Z,corr=0,0
    if (metallicity <LT['lZ'][0]):
        return 0.0; 
    if (metallicity>=LT['lZ'][LT['N_Z']-1]):
        return fromfit_4(mass, LT['windrate'][LT['N_Z']-1], LT['mup'][LT['N_Z']-1], 1, 0); 
    i_Z = get_Z_indfit(metallicity, LT['lZ'], LT['N_Z']);
    dlogZ = np.log10(metallicity/LT['lZ'][i_Z])/np.log10(LT['lZ'][i_Z+1]/LT['lZ'][i_Z]);
    return pow(10, (1-dlogZ)*fromfit_4(mass, LT['windrate'][i_Z], LT['mup'][i_Z], 0, 0) + dlogZ*fromfit_4(mass, LT['windrate'][i_Z+1], LT['mup'][i_Z+1], 0, 0)); 

def get_IR(mass,UnitEnergy_in_cgs=1.989e+43):
    logM = np.log10(mass)
    logL = -18.402 + 43.428557 * logM - 29.374363 * logM**2 + 6.40502 * logM**3
    return 10**logL *  4.807e50 / UnitEnergy_in_cgs


# Cooling Cal
k_B = 1.381e-16
# H2 #
o2p = 3.;
fo = 0.75;
fp = 0.25;
# CII Silva & Viegas 2002
A10CII = 2.3e-6
E10CII = 1.26e-14
g0CII = 2
g1CII = 4
# CI
g0CI = 1 ;
g1CI = 3 ;
g2CI = 5 ;
A10CI = 7.880e-08 ;
A20CI = 1.810e-14;
A21CI = 2.650e-07;
E10CI = 3.261e-15;
E20CI = 8.624e-15;
E21CI = 5.363e-15;
# OI
g0OI = 5 ;
g1OI = 3 ;
g2OI = 1 ;
A10OI = 8.910e-05 ;
A20OI = 1.340e-10 ;
A21OI = 1.750e-05 ;
E10OI = 3.144e-14 ;
E20OI = 4.509e-14 ;
E21OI = 1.365e-14 ;
zeta_CR = 10**-15.7   #s-1, Cosmic ray ionization rate
def get_CII_ALPHA_GAS(T):
    alpha = np.sqrt(T/6.67e-3)
    beta = np.sqrt(T/1.943e6)
    gamma = 0.7849+0.1597*np.exp(-49550/T)
    krr = 2.995e-9/(alpha*pow((1.+alpha),(1.-gamma))*pow((1.+beta),(1.+gamma)))
    kdr = pow(T,-1.5)*(6.346e-9*np.exp(-12.17/T)+9.793e-9*np.exp(-73.8/T)+1.634e-6*np.exp(-15230/T))
    return krr+kdr

def get_CII_ALPHA_GRAIN(T,NH,ne_cgs,X_FUV = 1,dgr = 1):
    #T in Kelvin, NH in cm-2, zfrac in solar, DZR absolute, X_FUV in Habing
    Av = NH * dgr / 1.87e21
    Phi = X_FUV*np.exp(-1.87*Av)*np.sqrt(T)/ne_cgs
    return dgr*4.558e-13/(1+6.089e-3*pow(Phi,1.128)*(1+433.1*pow(T,0.04845)*pow(Phi,-0.812-1.333e-4*np.log(T))))

def get_CII_BETA_CIIH2(T):
    return (2.31+0.99)*1e-13*pow(T,-1.3)*np.exp(-23/T)

def get_xCII(T, nh_cgs,NH_cgs, xC_tot, xhm, xe,y_fac, X_LW=1.7, X_FUV=1, zeta_CR = zeta_CR, dgr = 1):
    ne_cgs = nh_cgs * xe
    n_cgs = nh_cgs*(1+y_fac)
    tau_C = 1.6e-17 * NH_cgs * xC_tot
    tau_H2 = 2.8e-22 * NH_cgs * xhm
    fshield_C = np.exp(-tau_C)*np.exp(-tau_H2)/(1+tau_H2) #Tielens & Hollenbach (1985)
    zeta_pi_C = 3.43e-10 * X_LW * fshield_C + 520 * 2 * xhm * zeta_CR #Draine 1978, Gredel et al. 1987: CR-induced photoionization
    zeta_cr_C = 3.85*zeta_CR
    fCII = (zeta_pi_C+zeta_cr_C)/(zeta_pi_C+zeta_cr_C+get_CII_ALPHA_GAS(T)*ne_cgs+get_CII_ALPHA_GRAIN(T,NH_cgs,ne_cgs,X_FUV,dgr)*n_cgs+get_CII_BETA_CIIH2(T)*nh_cgs*xhm)
    return fCII * xC_tot

def get_xCO(nh_cgs, xhm, xC_tot, xO_tot, xCII, xOII, X_LW, dgr, zeta_CR):
    zeta16 = zeta_CR/1e-16
    n_CO_crit = pow(4e3*dgr*pow(zeta16,-2),pow(X_LW, 1/3))*(50*zeta16/pow(dgr,1.4))
    fCO = 2*xhm*(1-max(xCII/xC_tot,xOII/xO_tot))/(1+pow(n_CO_crit/nh_cgs,2))
    return fCO * xC_tot

def Solve2Level(q01,q10,A10):
    f1 = q01/(q01+q10+A10)
    return f1

def Solve3Level(q01, q10, q02, q20, q12, q21, A10, A20, A21):
    R10 = q10 + A10
    R20 = q20 + A20
    R21 = q21 + A21
    a0 = R10*R20 + R10*R21 + q12*R20
    a1 = q01*R20 + q01*R21 + R21*q02
    a2 = q02*R10 + q02*q12 + q12*q01
    de = a0 + a1 + a2
    f1 = a1 / de 
    f2 = a2 / de
    return f1, f2

def Lambda_CII158(nh_cgs, xCII, xh1, xhm, xe, T):
    nHI_cgs, nH2_cgs, ne_cgs = nh_cgs * xh1, nh_cgs * xhm, nh_cgs * xe
    # Draine (2011) ISM book eq (17.16) and (17.17)*/
    T2 = T/100.;
    k10e = 4.53e-8 * np.sqrt(1.0e4/T);
    k10HI = 7.58e-10 * pow(T2, 0.1281+0.0087*np.log10(T2));
    k10oH2 = 0;
    k10pH2 = 0;
    tmp = 0;
    if T < 500.:
        # fit in Wiesenfeld & Goldsmith 2014
        k10oH2 = (5.33 + 0.11*T2)*1.0e-10;
        k10pH2 = (4.43 + 0.33*T2)*1.0e-10;
    else:
    # Glover+ Jappsen 2007, for high temperature scales similar to HI
        tmp = pow(T, 0.07);
        k10oH2 = 3.74757785025e-10*tmp;
        k10pH2 = 3.88997286356e-10*tmp;
    k10H2 = k10oH2*fo + k10pH2*fp;
    
    
    q10 = k10e*ne_cgs + k10HI*nHI_cgs + k10H2*nH2_cgs
    q01 = (g1CII/g0CII) * q10 * np.exp( -E10CII/( k_B * T) );
    return xCII/nh_cgs*A10CII*E10CII*Solve2Level(q01, q10, A10CII);

def Lambda_CI370_CI610(nh_cgs, xCI, xh1, xhm, xe, T):
    nHI, nH2, ne = nh_cgs*xh1, nh_cgs*xhm, nh_cgs*xe
    #e collisional coefficents from Johnson, Burke, & Kingston 1987, JPhysB, 20, 2553
    T2 = T/100.;
    lnT2 = np.log(T2)
    lnT = np.log(T)

    #ke(u,l) = fac*gamma(u,l)/g(u)
    fac = 8.629e-8 * np.sqrt(1.0e4/T);
    if (T < 1.0e3) :
        lngamma10e = (((-6.56325e-4*lnT -1.50892e-2)*lnT + 3.61184e-1)*lnT -7.73782e-1)*lnT - 9.25141;
        lngamma20e = (((0.705277e-2*lnT - 0.111338)*lnT +0.697638)*lnT - 1.30743)*lnT -7.69735;
        lngamma21e = (((2.35272e-3*lnT - 4.18166e-2)*lnT +0.358264)*lnT - 0.57443)*lnT -7.4387;
    else:
        lngamma10e = (((1.0508e-1*lnT - 3.47620)*lnT + 4.2595e1)*lnT - 2.27913e2)*lnT + 4.446e2;
        lngamma20e = (((9.38138e-2*lnT - 3.03283)*lnT +3.61803e1)*lnT - 1.87474e2)*lnT +3.50609e2;
        lngamma21e = (((9.78573e-2*lnT - 3.19268)*lnT +3.85049e1)*lnT - 2.02193e2)*lnT +3.86186e2;

    k10e = fac * np.exp(lngamma10e) / g1CI;
    k20e = fac * np.exp(lngamma20e) / g2CI;
    k21e = fac * np.exp(lngamma21e) / g2CI;
    #HI collisional rates, Draine (2011) ISM book Appendix F Table F.6
    # NOTE: this is more updated than the LAMBDA database.
    k10HI = 1.26e-10 * pow(T2, 0.115+0.057*lnT2);
    k20HI = 0.89e-10 * pow(T2, 0.228+0.046*lnT2);
    k21HI = 2.64e-10 * pow(T2, 0.231+0.046*lnT2);
    #H2 collisional rates, Draine (2011) ISM book Appendix F Table F.6
    k10H2p = 0.67e-10 * pow(T2, -0.085+0.102*lnT2);
    k10H2o = 0.71e-10 * pow(T2, -0.004+0.049*lnT2);
    k20H2p = 0.86e-10 * pow(T2, -0.010+0.048*lnT2);
    k20H2o = 0.69e-10 * pow(T2, 0.169+0.038*lnT2);
    k21H2p = 1.75e-10 * pow(T2, 0.072+0.064*lnT2);
    k21H2o = 1.48e-10 * pow(T2, 0.263+0.031*lnT2);
    k10H2 = k10H2p*fp + k10H2o*fo;
    k20H2 = k20H2p*fp + k20H2o*fo;
    k21H2 = k21H2p*fp + k21H2o*fo;
    #The totol collisonal rates
    q10 = k10HI*nHI + k10H2*nH2 + k10e*ne;
    q20 = k20HI*nHI + k20H2*nH2 + k20e*ne;
    q21 = k21HI*nHI + k21H2*nH2 + k21e*ne;
    q01 = (g1CI/g0CI) * q10 * np.exp( -E10CI/(k_B*T) );
    q02 = (g2CI/g0CI) * q20 * np.exp( -E20CI/(k_B*T) );
    q12 = (g2CI/g1CI) * q21 * np.exp( -E21CI/(k_B*T) );
    f1, f2 = Solve3Level(q01, q10, q02, q20, q12, q21, A10CI, A20CI, A21CI)
    return  xCI/nh_cgs*( f1*A10CI*E10CI + f2*(A20CI*E20CI + A21CI*E21CI) )

def Lambda_OI63_OI146(nh_cgs, xOI, xh1, xhm, xe, T):
    nHI, nH2, ne = nh_cgs*xh1, nh_cgs*xhm, nh_cgs*xe
    #collisional rates from  Draine (2011) ISM book Appendix F Table F.6
    T2 = T/100;
    lnT2 = np.log(T2);
    #HI
    k10HI = 3.57e-10 * pow(T2, 0.419-0.003*lnT2); 
    k20HI = 3.19e-10 * pow(T2, 0.369-0.006*lnT2);
    k21HI = 4.34e-10 * pow(T2, 0.755-0.160*lnT2);
    #H2
    k10H2p = 1.49e-10 * pow(T2, 0.264+0.025*lnT2);
    k10H2o = 1.37e-10 * pow(T2, 0.296+0.043*lnT2);
    k20H2p = 1.90e-10 * pow(T2, 0.203+0.041*lnT2);
    k20H2o = 2.23e-10 * pow(T2, 0.237+0.058*lnT2);
    k21H2p = 2.10e-12 * pow(T2, 0.889+0.043*lnT2);
    k21H2o = 3.00e-12 * pow(T2, 1.198+0.525*lnT2);
    k10H2 = k10H2p*fp + k10H2o*fo;
    k20H2 = k20H2p*fp + k20H2o*fo;
    k21H2 = k21H2p*fp + k21H2o*fo;
    #e fit from Bell+1998
    k10e = 5.12e-10 * pow(T, -0.075);
    k20e = 4.86e-10 * pow(T, -0.026);
    k21e = 1.08e-14 * pow(T, 0.926);
    #total collisional rates
    q10 = k10HI*nHI + k10H2*nH2 + k10e * ne;
    q20 = k20HI*nHI + k20H2*nH2 + k20e * ne;
    q21 = k21HI*nHI + k21H2*nH2 + k21e * ne;
    q01 = (g1OI/g0OI) * q10 * np.exp( -E10OI/(k_B*T) );
    q02 = (g2OI/g0OI) * q20 * np.exp( -E20OI/(k_B*T) );
    q12 = (g2OI/g1OI) * q21 * np.exp( -E21OI/(k_B*T) );
    f1, f2 = Solve3Level(q01, q10, q02, q20, q12, q21, A10OI, A20OI, A21OI)
    return xOI/nh_cgs*( f1*A10OI*E10OI + f2*(A20OI*E20OI + A21OI*E21OI))

def Lambda_CO_WJ18(nh_cgs, NH_cgs, xhm, xCO, T):
    # divv: velocity divergence in km s-1 pc-1
    nhm_cgs =  nh_cgs * xhm
    X_CO = xCO/xhm
    pc_to_cm = 3.08567758128E+18
    R_scale = NH_cgs/nh_cgs/pc_to_cm
    divv = 1. * R_scale**0.5 / R_scale 
    Lambda_CO_LO = 2.16e-27 * nhm_cgs * pow(T, 3/2)
    Lambda_CO_HI = 2.21e-28 * (divv/X_CO) / nhm_cgs * pow(T,4)
    beta = 1.23 * pow(nhm_cgs, 0.0533) * pow(T, 0.164)
    Lambda = pow((pow(Lambda_CO_LO, -1/beta) + pow(Lambda_CO_HI, -1/beta)), -beta)
    
    #print(nh_cgs,Lambda,xCO,np.log10(X_CO/divv))
#     lambda_CO_LO = 5e-27 * (xCO/3e-4) * (T/10)**1.5 * (nh_cgs/1e3)
#     lambda_CO_HI = 2e-26 * divv * (nh_cgs/1e2)**-1 * (T/10)**4
#     beta = 1.23 * (nh_cgs/2)**0.0533 * T**0.164
#     return (lambda_CO_LO**(-1/beta) + lambda_CO_HI**(-1/beta))**(-beta)/nh_cgs
    return Lambda * xCO / nh_cgs

def Lambda(T, nh_cgs, NH_cgs, xh1, xhm, xe,y_fac, xC_tot = 1.6e-4, xO_tot = 3.2e-4, X_LW=1.7, X_FUV=1, zeta_CR = zeta_CR, dgr = 1):
    xh2 = 1 - xh1 - 2 * xhm
    xCII = get_xCII(T, nh_cgs, NH_cgs, xC_tot, xhm, xe,y_fac, X_LW, X_FUV, zeta_CR, dgr)
    xOII = xh2 * xO_tot
    xOI = xO_tot - xOII
    xCO = get_xCO(nh_cgs, xhm, xC_tot, xO_tot, xCII, xOII, X_LW, dgr, zeta_CR)
    xCI = xC_tot - xCII - xCO
    CII_Cooling = Lambda_CII158(nh_cgs, xCII, xh1, xhm, xe, T)
    CI_Cooling  = Lambda_CI370_CI610(nh_cgs, xCI, xh1, xhm, xe, T)
    OI_Cooling  = Lambda_OI63_OI146(nh_cgs, xOI, xh1, xhm, xe, T)
    CO_Cooling  = Lambda_CO_WJ18(nh_cgs, NH_cgs, xhm,  xCO, T)
    Lambda_metal = CII_Cooling + CI_Cooling + OI_Cooling + CO_Cooling
    return Lambda_metal

from scipy.optimize import brentq, newton, root_scalar
from scipy.interpolate import interpn, interp1d
from numba import njit
import numpy as np

def photoelectric_heating(X_FUV=1, nH=1, T=10, NH=0, Z=1):
    """
    Rate of photoelectric heating per H nucleus in erg/s.
    Weingartner & Draine 2001 prescription
    Grain charge parameter is a highly approximate fit vs. density - otherwise need to solve ionization.
    """
    grain_charge =  max((5e3/nH),50)
    c0=5.22; c1 =2.25; c2=0.04996; c3=0.0043; c4=0.147; c5=0.431; c6=0.692
    phi_PAH = 0.5
    #eps_PE = (c0+c1*T**c4)/(1+c2*grain_charge**c5 * (1+c5*grain_charge**c6))
    ne = nH * 2.7e-3*pow(T/1000,3/8)*pow(zeta_CR/1e-16,0.5)*pow(nH/10,-0.5)
    eps_PE = 4.87e-2/(1+4e-3*pow(X_FUV*np.sqrt(T)/(ne*phi_PAH),0.73)) + 3.65e-2*pow(T/1e4,0.7)/(1+2e-4*(X_FUV*np.sqrt(T)/(ne*phi_PAH)))
    sigma_FUV = 1e-21 * Z
    #print(X_FUV,nH,T, eps_PE, 1e-26 * X_FUV * eps_PE * np.exp(-NH*sigma_FUV) * Z)
    #return 1e-26 * X_FUV * eps_PE * np.exp(-NH*sigma_FUV) * Z
    #return 1.1e-25 * X_FUV /(1+3.2e-2*pow(X_FUV*pow(T/100,0.5)/ne*0.5,0.73))
    return 1.3e-24 * Z * X_FUV * eps_PE

def f_CO(nH=1, NH=1e21, T=10,X_FUV=1, Z=1):
    """Equilibrium fraction of C locked in CO, from Tielens 2005"""
    G0 = 1.7 * X_FUV * np.exp(-1e-21 * NH * Z)
    if nH > 10000*G0*340: return 1.
    x = (nH/(G0*340))**2*T**-0.5
    return x/(1+x)

def f_H2(nH=1, NH=1e21, X_FUV=1, Z=1):
    """Krumholz McKee Tumlinson 2008 prescription for fraction of neutral H in H_2 molecules"""
    surface_density_Msun_pc2 = NH * 1.1e-20  
    tau_UV = min(1e-21 * Z * NH,100.)
    G0 = 1.7 * X_FUV * np.exp(-tau_UV)
    chi = 71. * X_FUV / nH
    psi = chi * (1.+0.4*chi)/(1.+1.08731*chi)
    s = (Z + 1.e-3) * surface_density_Msun_pc2 / (1e-100 + psi)
    q = s * (125. + s) / (11. * (96. + s))
    fH2 = 1. - (1.+q*q*q)**(-1./3.)
    ind1 = (q < 0.2)
    ind2 = (q > 10)

    return fH2 * (1-ind1) * (1-ind2) + q*q*q * (1. - 2.*q*q*q/3.)/3. * ind1 + (1. - 1/q) * ind2

def H2_cooling(nH,NH,T,X_FUV,Z):
    """
    Glover & Abel 2008 prescription for H_2 cooling; accounts for H2-H2 and H2-HD collisions.
    Rate per H nucleus in erg/s.
    """
    f_molec = 0.5 * f_H2(nH,NH,X_FUV,Z)
    EXPmax = 90
    logT = np.log10(T)
    T3 = T/1000
    Lambda_H2_thick = (6.7e-19*np.exp(-min(5.86/T3,EXPmax)) + 1.6e-18*np.exp(-min(11.7/T3,EXPmax)) + 3.e-24*np.exp(-min(0.51/T3,EXPmax)) + 9.5e-22*pow(T3,3.76)*np.exp(-min(0.0022/(T3*T3*T3),EXPmax))/(1.+0.12*pow(T3,2.1))) / nH; #  super-critical H2-H cooling rate [per H2 molecule]
    Lambda_HD_thin = ((1.555e-25 + 1.272e-26*pow(T,0.77))*np.exp(-min(128./T,EXPmax)) + (2.406e-25 + 1.232e-26*pow(T,0.92))*np.exp(-min(255./T,EXPmax))) * np.exp(-min(T3*T3/25.,EXPmax)); #  optically-thin HD cooling rate [assuming all D locked into HD at temperatures where this is relevant], per molecule
    
    q = logT - 3.; Y_Hefrac=0.25; X_Hfrac=0.75; #  variable used below
    Lambda_H2_thin = max(nH-2.*f_molec,0) * X_Hfrac * np.power(10., max(-103. + 97.59*logT - 48.05*logT*logT + 10.8*logT*logT*logT - 0.9032*logT*logT*logT*logT , -50.)); #  sub-critical H2 cooling rate from H2-H collisions [per H2 molecule]; this from Galli & Palla 1998
    Lambda_H2_thin += Y_Hefrac * np.power(10., max(-23.6892 + 2.18924*q -0.815204*q*q + 0.290363*q*q*q -0.165962*q*q*q*q + 0.191914*q*q*q*q*q, -50.)); #  H2-He; often more efficient than H2-H at very low temperatures (<100 K); this and other H2-x terms below from Glover & Abel 2008
    Lambda_H2_thin += f_molec * X_Hfrac * np.power(10., max(-23.9621 + 2.09434*q -0.771514*q*q + 0.436934*q*q*q -0.149132*q*q*q*q -0.0336383*q*q*q*q*q, -50.)); #  H2-H2; can be more efficient than H2-H when H2 fraction is order-unity
    
    f_HD = min(0.00126*f_molec , 4.0e-5*nH)

    nH_over_ncrit = Lambda_H2_thin / Lambda_H2_thick
    Lambda_HD = f_HD * Lambda_HD_thin / (1. + f_HD/(f_molec+1e-10)*nH_over_ncrit) * nH 
    Lambda_H2 = f_molec * Lambda_H2_thin / (1. + nH_over_ncrit) * nH
    #return Lambda_H2 + Lambda_HD
    return Lambda_H2 

def CII_cooling(nH=1, Z=1, T=10, NH=1e21, X_FUV=1,prescription="Simple",zeta_CR=2e-16,y_fac = (1.0 - 0.76) / 4.0 / 0.76):    
    """Cooling due to atomic and/or ionized C. Uses either Hopkins 2022 FIRE-3 or simple prescription. Rate per H nucleus in erg/s."""
    if prescription=="Hopkins 2022 (FIRE-3)":
        return atomic_cooling_fire3(nH,NH,T,Z,X_FUV)
    if prescription=="Deng":
        xC_tot = 1.1e-4 * Z
        xO_tot = 2.2e-4 * Z
        xe = 2.7e-3*pow(T/1000,3/8)*pow(zeta_CR/1e-16,0.5)*pow(nH/10,-0.5) #Bialy19
        xhm = f_H2(nH, NH, X_FUV, Z)/2
        xh1 = 1-2*xhm
        X_Draine = X_FUV/1.7
        return nH * Lambda(T, nH, NH, xh1, xhm, xe,y_fac, xC_tot, xO_tot, X_Draine, X_Draine, zeta_CR, Z)
    if prescription=="Wolfire 2003":
        xhm = f_H2(nH, NH, X_FUV, Z)/2
        xh1 = 1-2*xhm
        xe = 2.7e-3*pow(T/1000,3/8)*pow(zeta_CR/1e-16,0.5)*pow(nH/10,-0.5) #Bialy19
        return nH * Lambda_W03(nH, xh1, xe, T, Z)
    T_CII = 91
    f_C = 1-f_CO(nH,NH,T,X_FUV,Z)
    xc = 1.1e-4
    return 8e-10 * 1.256e-14 * xc * np.exp(-T_CII/T) * Z * nH * f_C

def lyman_cooling(nH=1,T=1000):
    """Rate of Lyman-alpha cooling from Koyama & Inutsuka 2002 per H nucleus in erg/s. Actually a hard upper bound assuming xe ~ xH ~ xH+ ~ 1/2, see Micic 2013 for discussion."""
    return 2e-19 * np.exp(-1.184e5/T) * nH

def atomic_cooling_fire3(nH,NH,T,Z,X_FUV):
    """Cooling due to atomic and ionized C. Uses Hopkins 2022 FIRE-3 prescription. Rate per H nucleus in erg/s."""
    f = f_CO(nH,NH,T,X_FUV,Z)
    return 1e-27 * (0.47*T**0.15 * np.exp(-91/T) + 0.0208 * np.exp(-23.6/T)) * (1-f) * nH * Z

def get_tabulated_CO_coolingrate(T,NH,nH2):
    """Tabulated CO cooling rate from Omukai 2010, used for Gong 2017 implementation of CO cooling."""
    logT = np.log10(T)
    logNH = np.log10(NH)
    table = np.loadtxt("ismulator/coolingtables/omukai_2010_CO_cooling_alpha_table.dat")
    T_CO_table = np.log10(table[0,1:])
    NH_table = table[1:,0]
    alpha_table = table[1:,1:].T
    LLTE_table = np.loadtxt("ismulator/coolingtables/omukai_2010_CO_cooling_LLTE_table.dat")[1:,1:].T
    n12_table = np.loadtxt("ismulator/coolingtables/omukai_2010_CO_cooling_n12_table.dat")[1:,1:].T
    alpha = interpn((T_CO_table, NH_table), alpha_table, [[logT,logNH]],bounds_error=False,fill_value=None)
    LLTE = 10**-interpn((T_CO_table, NH_table), LLTE_table, [[logT,logNH]],bounds_error=False,fill_value=None)
    n12 = 10**interpn((T_CO_table, NH_table), n12_table, [[logT,logNH]],bounds_error=False,fill_value=None)
    L0 = 10**-np.interp(np.log10(T),T_CO_table,[24.77, 24.38, 24.21, 24.03, 23.89 ,23.82 ,23.42 ,23.13 ,22.91 ,22.63, 22.28])
    LM = (L0**-1 + nH2/LLTE + (1/L0)*(nH2/n12)**alpha * (1 - n12*L0/LLTE))**-1
    return LM

def CO_cooling(nH=1, T=10, NH=0,Z=1,X_FUV=1,divv=None,xCO=None,simple=False,prescription='Whitworth 2018'):
    """
    Rate of CO cooling per H nucleus in erg/s.
    Three prescriptions are implemented: Gong 2017, Whitworth 2018, and Hopkins 2022 (FIRE-3).
    Prescriptions that require a velocity gradient will assume a standard ISM size-linewidth relation by default, 
    unless div v is provided.2
    """
    fmol = f_CO(nH,NH,T,X_FUV,Z)
    pc_to_cm = 3.08567758128E+18
    if xCO is None:
        xCO = fmol * Z * 1.1e-4 * 2
    if divv is None:
        R_scale = NH/nH/pc_to_cm
        divv = 1. * R_scale**0.5 / R_scale # size-linewidth relation
    if prescription=="Gong 2017":
        n_H2 = fmol*nH/2
        neff = n_H2 + nH*2**0.5 * (2.3e-15/(3.3e-16*(T/1000)**-0.25)) # Eq. 34 from gong 2017, ignoring electrons 
        NCO = xCO * nH / (divv / pc_to_cm)
        LCO = get_tabulated_CO_coolingrate(T,NCO,neff)
        return LCO * xCO * n_H2
    elif prescription=='Hopkins 2022 (FIRE-3)':
        sigma_crit_CO=  1.3e19 * T / Z
        ncrit_CO=1.9e4 * T**0.5
        return 2.7e-31 * T**1.5 * (xCO/3e-4) * nH/(1 + (nH/ncrit_CO)*(1+NH/sigma_crit_CO)) #lambda_CO_HI)
    elif prescription=='Whitworth 2018':
        lambda_CO_LO = 5e-27 * (xCO/3e-4) * (T/10)**1.5 * (nH/1e3)
        lambda_CO_HI = 2e-26 * divv * (nH/1e2)**-1 * (T/10)**4
        beta = 1.23 * (nH/2)**0.0533 * T**0.164
        if simple: return np.min([lambda_CO_LO, lambda_CO_HI],axis=0)
        else: return (lambda_CO_LO**(-1/beta) + lambda_CO_HI**(-1/beta))**(-beta)

def CR_heating(T, nH, zeta_CR=2e-16, X_FUV = 1, Z = 1, NH=None):
    """Rate of cosmic ray heating in erg/s/H, just assuming 10eV per H ionization."""
    xe = 2.7e-3*pow(T/1000,3/8)*pow(zeta_CR/1e-16,0.5)*pow(nH/10,-0.5) #Bialy19
    q_H = (6.5+26.4*pow(xe/(xe+0.07),0.5)) * 1.6021766339999997e-12
    q_H2 = 10 
    xhm = f_H2(nH, NH, X_FUV, Z)/2
    if (nH>100) & (nH<4):
        q_H2 + 3*(np.log10(nH)-2)/4
    q_H2 *= 1.6021766339999997e-12
#     if NH is not None:
#         return (zeta_CR*(xhm*q_H2)+zeta_CR*(1-xhm)*q_H)/(1+(NH/1e21))
#     else:
#         return zeta_CR*(xhm*q_H2)+zeta_CR*(1-xhm)*q_H
    return (1-2*xhm)*1.6022e-12*zeta_CR + xhm*1.6022e-12*1.96*zeta_CR + (1.0 - 0.76) / 4.0 / 0.76 * 1.1 *1.6022e-12*zeta_CR
    
#     if NH is not None:
#         return 3e-27 * (zeta_CR / 2e-16)/(1+(NH/1e21))
#     else:
#         return 3e-27 * (zeta_CR / 2e-16)
    
@njit(fastmath=True,error_model='numpy')
def gas_dust_heating_coeff(T,Z,dust_coupling):
    """Coefficient alpha such that the gas-dust heat transfer is alpha * (T-T_dust)
    
    Uses Hollenbach & McKee 1979 prescription, assuming 10 Angstrom min grain size.
    """
    return 1.1e-32 * dust_coupling* Z * np.sqrt(T) * (1-0.8*np.exp(-75/T)) 

@njit(fastmath=True,error_model='numpy')
def dust_gas_cooling(nH=1,T=10,Tdust=20,Z=1,dust_coupling=1):
    """
    Rate of heat transfer from gas to dust in erg/s per H
    """
    return nH * gas_dust_heating_coeff(T,Z,dust_coupling) * (Tdust-T)  #3e-26 * Z * (T/10)**0.5 * (Tdust-T)/10 * (nH/1e6)

def compression_heating(nH=1,T=10):
    """Rate of compressional heating per H in erg/s, assuming freefall collapse (e.g. Masunaga 1998)"""
    return 1.2e-27 * np.sqrt(nH/1e6) * (T/10) # ceiling here to get reasonable results in diffuse ISM

def turbulent_heating(sigma_GMC=100., M_GMC=1e6):
    """
    Rate of tubulent dissipation per H in erg/s, assuming a turbulent GMC with virial parameter=1 of a certain surface density and mass.
    
    Note that much of the cooling can take place in shocks that are way out of equilibrium, so this doesn't necessarily capture the full effect.
    """
    return 5e-27 * (M_GMC/1e6)**0.25 * (sigma_GMC/100)**(1.25)

def dust_temperature(nH=1,T=10,Z=1,NH=0, X_FUV=1, X_OPT=1, z=0,beta=2,dust_coupling=1):
    """
    Equilibrium dust temperature obtained by solving the dust energy balance equation accounting for absorption, emission, and gas-dust heat transfer.    
    """
    abs = dust_absorption_rate(NH,Z,X_FUV,X_OPT,z, beta)
    sigma_IR_0 = 2e-25
    Tdust_guess = 10*(abs / (sigma_IR_0 * Z * 2.268))**(1./(4+beta))
    Tdust_guess = max(Tdust_guess, (abs/(4.5e-23*Z*2.268))**0.25)
    Tdust_guess = max(Tdust_guess, T - 2.268 * sigma_IR_0 * Z * (min(T,150)/10)**beta * (T/10)**4 /  (gas_dust_heating_coeff(T,Z,dust_coupling)*nH))

    func = lambda dT: net_dust_heating(dT, nH,T,NH,Z,X_FUV,X_OPT,z,beta,abs,dust_coupling) # solving for the difference T - Tdust since that's what matters for dust heating
    result = root_scalar(func, x0 = Tdust_guess-T,x1 =(Tdust_guess*1.1 - T), method='secant',xtol=1e-5)#,rtol=1e-3,xtol=1e-4*T)
    Tdust = T+result.root
    if not result.converged:
        func = lambda logT: net_dust_heating(10**logT - T, nH,T,NH,Z,X_FUV,X_OPT,z,beta,abs,dust_coupling)
        result = root_scalar(func, bracket=[-1,8], method='brentq')
        Tdust = 10**result.root
    return Tdust

@njit(fastmath=True,error_model='numpy')
def net_dust_heating(dT,nH,T,NH,Z=1,X_FUV=1,X_OPT=1,z=0, beta=2, absorption=-1,dust_coupling=1):
    """Derivative of the dust energy in the dust energy equation, solve this = 0 to get the equilibrium dust temperature."""
    Td = T + dT
    sigma_IR_0 = 2e-25
    sigma_IR_emission = sigma_IR_0 * Z * (min(Td,150)/10)**beta # dust cross section per H in cm^2
    lambdadust_thin = 2.268 * sigma_IR_emission * (Td/10)**4
    lambdadust_thick = 2.268 * (Td/10)**4 / (NH+1e-100)
    psi_IR = 1/(1/lambdadust_thin + 1/lambdadust_thick) #interpolates the lower envelope of the optically-thin and -thick limits
    lambda_gd = dust_gas_cooling(nH,T,Td,Z,dust_coupling)
    
    if absorption < 0:
        absorption = dust_absorption_rate(NH,Z,X_FUV,X_OPT,z, beta)
    return absorption - lambda_gd - psi_IR

@njit(fastmath=True,error_model='numpy')
def dust_absorption_rate(NH,Z=1,X_FUV=1,X_OPT=1,z=0, beta=2):
    """Rate of radiative absorption by dust, per H nucleus in erg/s."""
    T_CMB = 2.73*(1+z)
    X_OPT_eV_cm3 = X_OPT * 0.54
    X_IR_eV_cm3 = X_OPT * 0.39
    X_FUV_eV_cm3 = X_FUV * 0.041
    T_IR=max(20,3.8*(X_OPT_eV_cm3+X_IR_eV_cm3)**0.25)  # assume 20K dust emission or the blackbody temperature of the X_FUV energy density, whichever is greater

    sigma_UV = 1e-21 * Z
    sigma_OPT = 3e-22 * Z
    sigma_IR_0 = 2e-25
    sigma_IR_CMB = sigma_IR_0 * Z * (min(T_CMB,150)/10)**beta
    sigma_IR_ISRF = sigma_IR_0 * Z * (min(T_IR,150)/10)**beta
    
    tau_UV = min(sigma_UV * NH,100)
    gamma_UV = X_FUV * np.exp(-tau_UV) * 5.965e-25 * Z

    tau_OPT = min(sigma_OPT * NH,100)
    gamma_OPT = X_OPT * 7.78e-24 * np.exp(-tau_OPT) * Z
    gamma_IR =  2.268 * sigma_IR_CMB * (T_CMB/10)**4 + 0.048 * (X_IR_eV_cm3 + X_OPT_eV_cm3 * (-np.expm1(-tau_OPT)) + X_FUV_eV_cm3 * (-np.expm1(-tau_UV))) * sigma_IR_ISRF
    return gamma_IR + gamma_UV + gamma_OPT

all_processes = "CR Heating", "Lyman cooling", "Photoelectric", "CII Cooling", "CO Cooling", "Dust-Gas Coupling", "Grav. Compression", "H_2 Cooling", "Turb. Dissipation"

def net_heating(T=10, nH=1, NH=0, X_FUV=1,X_OPT=1,Z=1, z=0, divv=None, zeta_CR=2e-16, Tdust=None,jeans_shielding=False,dust_beta=2.,dust_coupling=1,sigma_GMC=100,M_GMC=1e6, processes=all_processes, attenuate_cr=True,co_prescription="Whitworth 2018",cii_prescription="Hopkins 2022 (FIRE-3)",y_fac=(1.0 - 0.76) / 4.0 / 0.76):
    if jeans_shielding:
        lambda_jeans = 8.1e19 * nH**-0.5 * (T/10)**0.5
        NH = np.max([nH*lambda_jeans*jeans_shielding, NH],axis=0)
    if Tdust==None:
        Tdust = dust_temperature(nH,T,Z,NH,X_FUV,X_OPT,z,dust_beta,dust_coupling * ("Dust-Gas Coupling" in processes))
    rate = 0
    for process in processes:
        if process == "CR Heating": rate += CR_heating(T, nH, zeta_CR, X_FUV, Z, NH=NH*attenuate_cr)
        if process == "Lyman cooling": rate -= lyman_cooling(nH,T)
        if process == "Photoelectric": rate += photoelectric_heating(X_FUV, nH,T, NH, Z)
        if process == "CII Cooling": rate -= CII_cooling(nH, Z, T, NH, X_FUV,prescription=cii_prescription,zeta_CR = zeta_CR,y_fac = y_fac)
        if process == "CO Cooling": rate -= CO_cooling(nH,T,NH,Z,X_FUV,divv,prescription=co_prescription)
        if process == "Dust-Gas Coupling": rate += dust_gas_cooling(nH,T,Tdust,Z,dust_coupling)
        if process == "H_2 Cooling": rate -= H2_cooling(nH,NH,T,X_FUV,Z)
        if process == "Grav. Compression": rate += compression_heating(nH,T)
        if process == "Turb. Dissipation": rate += turbulent_heating(sigma_GMC=sigma_GMC,M_GMC=M_GMC)
        
    return rate
def equilibrium_temp(nH=1, NH=0, X_FUV=1,X_OPT=1, Z=1, z=0, divv=None, zeta_CR=2e-16, Tdust=None,jeans_shielding=False,
                     dust_beta=2.,dust_coupling=1,sigma_GMC=100.,M_GMC=1e6, processes=all_processes,attenuate_cr=True,
                     co_prescription="Whitworth 2018",cii_prescription="Hopkins 2022 (FIRE-3)",y_fac=(1.0 - 0.76) / 4.0 / 0.76,return_Tdust=True,T_guess=None):
    
    if NH==0: NH=1e18
    params = nH, NH, X_FUV,X_OPT, Z, z, divv, zeta_CR, Tdust, jeans_shielding, dust_beta, dust_coupling, sigma_GMC,M_GMC,processes, attenuate_cr, co_prescription, cii_prescription,y_fac
    func = lambda logT: net_heating(10**logT, *params) # solving vs logT converges a bit faster
    
    use_brentq = True
    if T_guess is not None: # we have an initial guess that is supposed to be close (e.g. previous grid point)
        T_guess2 = T_guess * 1.01
        result = root_scalar(func, x0=np.log10(T_guess),x1=np.log10(T_guess2), method='secant',rtol=1e-3) #,rtol=1e-3,xtol=1e-4*T)
        if result.converged:
            T = 10**result.root; use_brentq = False

    if use_brentq: 
        try:
            T = 10**brentq(func, -1,5,rtol=1e-3,maxiter=500)
        except:
            try:
                T = 10**brentq(func, -1,10,rtol=1e-3,maxiter=500)
            except:
                raise("Couldn't solve for temperature! Try some other parameters.")

    if return_Tdust:
        Tdust = dust_temperature(nH,T,Z,NH,X_FUV,X_OPT,z,dust_beta,dust_coupling*("Dust-Gas Coupling" in processes))
        return T, Tdust
    else:
        return T

def equilibrium_temp_grid(nH, NH, X_FUV=1, X_OPT=1, Z=1, z=0, divv=None, zeta_CR=2e-16, Tdust=None,jeans_shielding=False,
    dust_beta=2.,dust_coupling=1,sigma_GMC=100.,M_GMC=1e6, processes=all_processes,attenuate_cr=True,return_Tdust=False,
    co_prescription="Whitworth 2018",cii_prescription="Hopkins 2022 (FIRE-3)",y_fac=(1.0 - 0.76) / 4.0 / 0.76):
    
    params = X_FUV, X_OPT, Z, z, divv, zeta_CR, Tdust, jeans_shielding, dust_beta, dust_coupling, sigma_GMC, M_GMC, processes, attenuate_cr, co_prescription, cii_prescription, y_fac
    Ts = []
    Tds = []
    T_guess = None
    for i in prange(len(nH)): # we do a pass on the grid where we use previously-evaluated temperatures to get good initial guesses for the next grid point
        if i==1:
            T_guess = Ts[-1]
        elif i>1:
            T_guess = 10**interp1d(np.log10(nH[:i]),np.log10(Ts),fill_value="extrapolate")(np.log10(nH[i])) # guess using linear extrapolation in log space
        sol = equilibrium_temp(nH[i],NH[i],*params,return_Tdust=return_Tdust,T_guess=T_guess)
        if return_Tdust:
            T, Tdust = sol
            Ts.append(T)
            Tds.append(Tdust)
        else:
            Ts.append(sol)
    if return_Tdust:
        return np.array(Ts), np.array(Tds)
    else:
        return np.array(Ts)
def get_equilibrium_curve(snap,mask_sf,n,rho0,M):
    #NH2 = np.mean((snap['0_H2_Fraction'][mask_sf]*snap['0_Density'][mask_sf]/2/c.m_p*snap['0_Diameters'][mask_sf]).to('cm-2').value)
    NH2 = np.median((2*snap['0_H2_Fraction'][mask_sf]*snap['0_Density'][mask_sf]/c.m_p*snap['0_Diameters'][mask_sf]).to('cm-2').value)
    #NH2 = (np.sum(snap['0_H2_Fraction'][mask_sf]*snap['0_Masses'][mask_sf])/2/c.m_p/np.sum(3*snap['0_Volume'][mask_sf]/4/np.pi)**(2/3)/np.pi).to('cm-2').value
    #Z = 0.03288415123139433             #Metalicity in solar
    Z = (np.sum(snap['0_GFM_MetallicityTimesMasses'][mask_sf])/np.sum(snap['0_Masses'][mask_sf])/0.012892950745).value
    y_fac = (np.sum(snap['0_GFM_Metals'][mask_sf,1]*snap['0_Masses'][mask_sf])/np.sum(snap['0_Masses'][mask_sf])).value
    z = 0                 #redshift
    Habing = u.Unit(5.29e-14*u.erg*u.cm**(-3))
    Draine = 1.71*Habing
    #X_FUV = (np.median(snap['0_PhotonDensity'][mask_sf,1]*8.45*u.eV+snap['0_PhotonDensity'][mask_sf,2]*12.26*u.eV)).to(Habing).value*1e63         #Habing, FUV radiation field strength
    #X_OPT = (np.median(snap['0_PhotonDensity'][mask_sf,0])*3.67*u.eV).to(Habing).value*1e63        #IR-Opt radiation field strength
    X_FUV = (np.sum((snap['0_PhotonDensity'][mask_sf,1]*8.45*u.eV+snap['0_PhotonDensity'][mask_sf,2]*12.26*u.eV)*snap['0_Volume'][mask_sf])/np.sum(snap['0_Volume'][mask_sf])).to(Habing).value*1e63         #Habing, FUV radiation field strength
    X_OPT = (np.sum(snap['0_PhotonDensity'][mask_sf,0]*3.67*u.eV*snap['0_Volume'][mask_sf])/np.sum(snap['0_Volume'][mask_sf])).to(Habing).value*1e63        #IR-Opt radiation field strength
    #Tdust = np.median(snap['0_DustTemperature'][mask_sf]).to(u.K).value
    Tdust = (np.sum(snap['0_DustTemperature'][mask_sf]*snap['0_Masses'][mask_sf])/np.sum(snap['0_Masses'])).to(u.K).value
    NH_alpha = 0.3        #Column density scaling: $\alpha$ where $N_{\rm H}=N_{\rm H,0}\left(n_{\rm H}/100\rm cm^{-3}\right)^{\alpha}$
    fJ = 0.25             #Ratio of shielding length floor to Jeans wavelength
    dust_beta = 2.        #Dust spectral index
    dust_coeff = 10**0.   #dust-gas coupling coefficient
    sigma_GMC = 10**2.    #turb. dissipation'
    attenuate_cr = True   #Cosmic ray attenuation
    NH = NH2 * (n/1e2)**NH_alpha
    
    processes_Deng = ["CR Heating", "Lyman cooling", "Photoelectric", "CII Cooling", "H_2 Cooling", "Dust-Gas Coupling","Grav. Compression","CO Cooling","Turb. Dissipation"]
    Teq_Deng = equilibrium_temp_grid(n, NH, X_FUV=X_FUV, X_OPT=X_OPT, Z=Z, z=z, divv=None, zeta_CR=1.78e-16, Tdust=Tdust,jeans_shielding=False,
    dust_beta=2.,dust_coupling=1,sigma_GMC=rho0.value,M_GMC=M.value, processes=processes_Deng,attenuate_cr=attenuate_cr,return_Tdust=False,
    co_prescription="Whitworth 2018",cii_prescription='Deng',y_fac = y_fac)
    return Teq_Deng
def get_equilibrium_curve_binnedbyn(snap,mask_sf,n,rho0,M, z=0, divv=None, zeta_CR=2e-16, jeans_shielding=False,dust_beta=2.,dust_coupling=1,attenuate_cr=True,return_Tdust=False,co_prescription="Whitworth 2018",cii_prescription="Hopkins 2022 (FIRE-3)"):

    Habing = u.Unit(5.29e-14*u.erg*u.cm**(-3))
    sigma_GMC=rho0.value;M_GMC=M.value
    X_FUV_global = (np.sum((snap['0_PhotonDensity'][mask_sf,1]*8.45*u.eV + snap['0_PhotonDensity'][mask_sf,2]*12.26*u.eV)*snap['0_Volume'][mask_sf]) / np.sum(snap['0_Volume'][mask_sf])).to(Habing).value*1e63
    X_OPT_global = (np.sum(snap['0_PhotonDensity'][mask_sf,0]*3.67*u.eV*snap['0_Volume'][mask_sf]) / np.sum(snap['0_Volume'][mask_sf])).to(Habing).value*1e63
    Tdust_global = (np.sum(snap['0_DustTemperature'][mask_sf]*snap['0_Masses'][mask_sf]) / np.sum(snap['0_Masses'][mask_sf])).to(u.K).value
    Z_global = np.sum(snap['0_GFM_MetallicityTimesMasses'][mask_sf]) / np.sum(snap['0_Masses'][mask_sf]) / 0.012892950745
    y_fac_global = (np.sum(snap['0_GFM_Metals'][mask_sf,1]*snap['0_Masses'][mask_sf]) / np.sum(snap['0_Masses'][mask_sf])).value

    logn = np.log10(n)
    edges_log = np.empty(len(n) + 1)
    edges_log[1:-1] = 0.5 * (logn[:-1] + logn[1:])
    edges_log[0] = logn[0] - (edges_log[1] - logn[0])
    edges_log[-1] = logn[-1] + (logn[-1] - edges_log[-2])
    edges = 10**edges_log
    numdens = snap['0_NumberDensity'].to('cm-3').value
    vol = snap['0_Volume']
    mass = snap['0_Masses']
    phot = snap['0_PhotonDensity']
    dustT = snap['0_DustTemperature']
    processes_Deng = ["CR Heating", "Lyman cooling", "Photoelectric", "CII Cooling", "H_2 Cooling", "Dust-Gas Coupling","Grav. Compression","CO Cooling","Turb. Dissipation"]
    Ts =  np.zeros(len(n))
    Tds = np.zeros(len(n))
    for i in prange(len(n)):
        lo, hi = edges[i], edges[i+1]
        pmask = np.logical_and(numdens >= lo, numdens < hi)
        if pmask.sum() == 0:
            X_FUV = X_FUV_global
            X_OPT = X_OPT_global
            Tdust = Tdust_global
            Z = Z_global.value
            y_fac = y_fac_global
            NH = np.median((snap['0_HI_Fraction'][mask_sf] * snap['0_Density'][mask_sf] / c.m_p * snap['0_Diameters'][mask_sf]).to('cm-2').value)
        else:
            num = np.sum((phot[pmask,1]*8.45*u.eV + phot[pmask,2]*12.26*u.eV) * vol[pmask])
            den = np.sum(vol[pmask])
            X_FUV = (num/den).to(Habing).value*1e63
            numo = np.sum(phot[pmask,0]*3.67*u.eV * vol[pmask])
            X_OPT = (numo/den).to(Habing).value*1e63
            Tdust = (np.sum(dustT[pmask]*mass[pmask]) / np.sum(mass[pmask])).to(u.K).value
            Z = (np.sum(snap['0_GFM_MetallicityTimesMasses'][pmask]) / np.sum(mass[pmask]) / 0.012892950745).value
            y_fac = (np.sum(snap['0_GFM_Metals'][pmask,1]*mass[pmask]) / np.sum(mass[pmask])).value
            NH = (np.sum(snap['0_HI_Fraction'][pmask] * snap['0_Masses'][pmask]/ c.m_p)/np.sum(np.pi*snap['0_Diameters'][pmask]**2/4)).to('cm-2').value
        params = X_FUV, X_OPT, Z, z, divv, zeta_CR, Tdust, jeans_shielding, dust_beta, dust_coupling, sigma_GMC, M_GMC, processes_Deng, attenuate_cr, co_prescription, cii_prescription, y_fac
        T_guess = None
        if i==1:
            T_guess = Ts[i-1]
        elif i>1:
            T_guess = 10**interp1d(np.log10(n[:i]),np.log10(Ts[:i]),fill_value="extrapolate")(np.log10(n[i])) # guess using linear extrapolation in log space
        sol = equilibrium_temp(n[i],NH,*params,return_Tdust=return_Tdust,T_guess=T_guess)
        if return_Tdust:
            T, Tdust = sol
            Ts[i] = T
            Tds[i] = Tdust
        else:
            Ts[i] = sol
    if return_Tdust:
        return Ts, Tds
    else:
        return Ts
def process_cal( nHs, NHs,T=100, X_FUV=1,X_OPT=1,Z=1, z=0, divv=None, zeta_CR=2e-16, Tdust=None,jeans_shielding=False,dust_beta=2.,dust_coupling=1,sigma_GMC=100,M_GMC=1e6, processes=all_processes, attenuate_cr=True,co_prescription="Whitworth 2018",cii_prescription="Hopkins 2022 (FIRE-3)",y_fac=(1.0 - 0.76) / 4.0 / 0.76):
    process_rate=np.zeros((len(processes),len(nHs)))
    for i in prange(len(nHs)):
        nH = nHs[i]
        NH = NHs[i]
        if jeans_shielding:
            lambda_jeans = 8.1e19 * nH**-0.5 * (T/10)**0.5
            NH = np.max([nH*lambda_jeans*jeans_shielding, NH],axis=0)
        if Tdust==None:
            Tdust = dust_temperature(nH,T,Z,NH,X_FUV,X_OPT,z,dust_beta,dust_coupling * ("Dust-Gas Coupling" in processes))
        for j in range(len(processes)):
            process = processes[j]
            if process == "CR Heating": process_rate[j,i] = CR_heating(T, nH, zeta_CR, X_FUV, Z, NH=NH*attenuate_cr)
            if process == "Lyman cooling": process_rate[j,i] = lyman_cooling(nH,T)
            if process == "Photoelectric": process_rate[j,i] = photoelectric_heating(X_FUV, nH,T, NH, Z)
            if process == "CII Cooling": process_rate[j,i] = CII_cooling(nH, Z, T, NH, X_FUV,prescription=cii_prescription,zeta_CR = zeta_CR,y_fac=y_fac)
            if process == "CO Cooling": process_rate[j,i] = CO_cooling(nH,T,NH,Z,X_FUV,divv,prescription=co_prescription)
            if process == "Dust-Gas Coupling": process_rate[j,i] = dust_gas_cooling(nH,T,Tdust,Z,dust_coupling)
            if process == "H_2 Cooling": process_rate[j,i] = H2_cooling(nH,NH,T,X_FUV,Z)
            if process == "Grav. Compression": process_rate[j,i] = compression_heating(nH,T)
            if process == "Turb. Dissipation": process_rate[j,i] = turbulent_heating(sigma_GMC=sigma_GMC,M_GMC=M_GMC)
    return process_rate
def process_single_cal( nH, NH,T=100, X_FUV=1,X_OPT=1,Z=1, z=0, divv=None, zeta_CR=2e-16, Tdust=None,jeans_shielding=False,dust_beta=2.,dust_coupling=1,sigma_GMC=100,M_GMC=1e6, processes=all_processes, attenuate_cr=True,co_prescription="Whitworth 2018",cii_prescription="Hopkins 2022 (FIRE-3)",y_fac=(1.0 - 0.76) / 4.0 / 0.76):
    process_rate=np.zeros(len(processes))
    if jeans_shielding:
        lambda_jeans = 8.1e19 * nH**-0.5 * (T/10)**0.5
        NH = np.max([nH*lambda_jeans*jeans_shielding, NH],axis=0)
    if Tdust==None:
        Tdust = dust_temperature(nH,T,Z,NH,X_FUV,X_OPT,z,dust_beta,dust_coupling * ("Dust-Gas Coupling" in processes))
    for j in range(len(processes)):
        process = processes[j]
        if process == "CR Heating": process_rate[j] = CR_heating(T, nH, zeta_CR, X_FUV, Z, NH=NH*attenuate_cr)
        if process == "Lyman cooling": process_rate[j] = lyman_cooling(nH,T)
        if process == "Photoelectric": process_rate[j] = photoelectric_heating(X_FUV, nH,T, NH, Z)
        if process == "CII Cooling": process_rate[j] = CII_cooling(nH, Z, T, NH, X_FUV,prescription=cii_prescription,zeta_CR = zeta_CR,y_fac=y_fac)
        if process == "CO Cooling": process_rate[j] = CO_cooling(nH,T,NH,Z,X_FUV,divv,prescription=co_prescription)
        if process == "Dust-Gas Coupling": process_rate[j] = dust_gas_cooling(nH,T,Tdust,Z,dust_coupling)
        if process == "H_2 Cooling": process_rate[j] = H2_cooling(nH,NH,T,X_FUV,Z)
        if process == "Grav. Compression": process_rate[j] = compression_heating(nH,T)
        if process == "Turb. Dissipation": process_rate[j] = turbulent_heating(sigma_GMC=sigma_GMC,M_GMC=M_GMC)
    return process_rate
def process_loop(n,NH,X_FUV,X_OPT,T,Tdust,processes,Z,Vol,y_fac,divv):
    mean_coolingrate = np.zeros(len(processes))
    for ni in prange(len(n)):
        process_rate = process_single_cal(nH=n[ni],NH=NH[ni],X_FUV=X_FUV[ni],X_OPT=X_OPT[ni],T=T[ni],Tdust=Tdust[ni],divv=divv[ni],processes=processes,Z=Z,y_fac=y_fac[ni])
        mean_coolingrate += process_rate*n[ni]**2*Vol[ni]
        #mean_coolingrate += process_rate
    mean_coolingrate /= np.sum(n*Vol)
    return mean_coolingrate