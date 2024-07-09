import numpy as np

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