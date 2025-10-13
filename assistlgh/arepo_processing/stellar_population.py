import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import quad

def linear_interp(l,r,lv,rv,x):
    return (rv-lv)/(r-l)*(x-l)+lv
    #return (x-r)/(l-r)*lv-rv*(x-l)/(l-r)
def process_single_element(func, value):
    if callable(func):
        return func(value)
    else:
        return func
def gamma_largeq(m1,p):
    logp=np.log10(p).reshape(-1)
    gamma_largeq = np.zeros(logp.shape)
    if np.logical_and(m1>=0.8,m1<1.2):
        boundary = np.array([0.2,5,8])
        indices = np.digitize(logp, boundary).reshape(-1)
        func_list = [-0.5,-0.5,lambda logp:-0.5-0.3*(logp-5),-0.5-0.3*(8-5)]
    elif m1>6:
        boundary = np.array([0,1,2,4,8])
        indices = np.digitize(logp, boundary).reshape(-1)
        func_list=[-0.5,-0.5, lambda logp:-0.5-0.9*(logp-1),lambda logp:-1.4-0.3*(logp-2),-2,-2]
    elif m1==3.5:
        boundary = np.array([0.2,1,4.5,6.5,8])
        indices = np.digitize(logp, boundary).reshape(-1)
        func_list = [-0.5,-0.5,lambda logp:-0.5-0.2*(logp-1),lambda logp:-1.2-0.4*(logp-4.5),-2,-2]
    elif m1>=1.2 and m1<3.5:
        l=1.2;r=3.5
        boundary = np.array([0.2,1,4.5,5,6.5,8])
        indices = np.digitize(logp, boundary).reshape(-1)
        #(m1-r)/(l-r)*lv-rv*(m1-l)/(l-r)
        func_list = [(m1-r)/(l-r)*(-0.5)-(-0.5)*(m1-l)/(l-r),
                        (m1-r)/(l-r)*(-0.5)-(-0.5)*(m1-l)/(l-r),
                        lambda logp:(m1-r)/(l-r)*(-0.5)-(-0.5-0.2*(logp-1))*(m1-l)/(l-r),
                        lambda logp: (m1-r)/(l-r)*(-0.5)-(-1.2-0.4*(logp-4.5))*(m1-l)/(l-r),
                        lambda logp:(m1-r)/(l-r)*(-0.5-0.3*(logp-5))-(-1.2-0.4*(logp-4.5))*(m1-l)/(l-r),
                        lambda logp: (m1-r)/(l-r)*(-0.5-0.3*(logp-5))-(-2)*(m1-l)/(l-r),
                    (m1-r)/(l-r)*(-0.5-0.3*(8-5))-(-2)*(m1-l)/(l-r)
                    ]
    elif m1>3.5 and m1<=6:
        l=3.5;l_r=6-3.5
        boundary = np.array([0,0.2,1,2,4,4.5,6.5,8])
        indices = np.digitize(logp, boundary).reshape(-1)
        func_list = [-0.5,
                        -0.5,
                        -0.5,
                        lambda logp:-0.7*(logp-1)/l_r*(m1-l)-0.5-0.2*(logp-1),
                        lambda logp:(-0.1*logp-0.5)/l_r*(m1-l)-0.5-0.2*(logp-1),
                        lambda logp:(-1.5+0.2*(logp-1))/l_r*(m1-l)-0.5-0.2*(logp-1),
                        lambda logp:(-0.8+0.4*(logp-4.5))/l_r*(m1-l)-1.2-0.4*(logp-4.5),
                        -2,
                    -2]
    for i,index in enumerate(indices):
        gamma_largeq[i]=process_single_element(func_list[index],logp[i])
    return gamma_largeq
def gamma_smallq(m1,p):
    logp=np.log10(p).reshape(-1)
    gamma_smallq = np.zeros(logp.shape)
    if np.logical_and(m1>=0.8,m1<1.2):
        boundary = np.array([0.2,8])
        indices = np.digitize(logp, boundary).reshape(-1)
        func_list = [0.3,0.3,0.3]
    elif m1>6:
        boundary = np.array([0.2,1,3,5.6,8])
        indices = np.digitize(logp, boundary).reshape(-1)
        func_list = [0.1,0.1,lambda logp:0.1-0.15*(logp-1),lambda logp:-0.2-0.5*(logp-3),-1.5,-1.5]
    elif m1==3.5:
        boundary = np.array([0.2,2.5,5.5,8])
        indices = np.digitize(logp, boundary).reshape(-1)
        func_list = [0.2,0.2,lambda logp:0.2-0.3*(logp-2.5),lambda logp:-0.7-0.2*(logp-5.5),-0.7-0.2*(8-5.5)]
    elif m1>=1.2 and m1<3.5:
        l=1.2;r=3.5
        boundary = np.array([0.2,2.5,5.5,8])
        indices = np.digitize(logp, boundary).reshape(-1)
        #(m1-r)/(l-r)*lv-rv*(m1-l)/(l-r)
        func_list = [-0.1/(r-l)*(m1-l)+0.3,
                        -0.1/(r-l)*(m1-l)+0.3,
                        lambda logp:(-0.1-0.3*(logp-2.5))/(r-l)*(m1-l)+0.3,
                        lambda logp: (-1-0.2*(logp-5.5))*(m1-l)/(r-l)+0.3,
                    (-1-0.2*(8-5.5))*(m1-l)/(r-l)+0.3
                        ]
    elif m1>3.5 and m1<=6:
        l=3.5;r_l=6-3.5
        boundary = np.array([0.2,1,2.5,3,5.5,5.6,8])
        indices = np.digitize(logp, boundary).reshape(-1)
        func_list = [-0.1/2.5*(m1-l)+0.2,
                        -0.1/2.5*(m1-l)+0.2,
                        lambda logp: (-0.1-0.15*(logp-1))/2.5*(m1-l)+0.2,
                        lambda logp:(-0.7+0.15*logp)/2.5*(m1-l)+0.2-0.3*(logp-2.5),
                        lambda logp: (0.35-0.2*logp)/2.5*(m1-l)+0.2-0.3*(logp-2.5),
                        lambda logp:(0.9-0.3*logp)/2.5*(m1-l)-0.7-0.2*(logp-5.5),
                        lambda logp:(-1.9+0.2*logp)/2.5*(m1-l)-0.7-0.2*(logp-5.5),
                    (-1.9+0.2*8)/2.5*(m1-l)-0.7-0.2*(8-5.5)
                        ]
    for i,index in enumerate(indices):
        gamma_smallq[i]=process_single_element(func_list[index],logp[i])
    return gamma_smallq
def Ftwins(m1,p):
    logp=np.log10(p).reshape(-1)
    m1 = np.array([m1]).reshape(-1)
    #if q>0.95: #Ftwins
    F = np.zeros((len(m1),len(logp)))
    for m_index,mm in enumerate(m1):
        if mm<=6.5:
            logpt=8-mm
        else:
            logpt=1.5
        F[m_index,logp<1] =  np.full(len(F[m_index,logp<1]),0.3-0.15*np.log10(mm))
        mask = np.logical_and(logp>=1,logp<logpt)
        F[m_index,mask] +=  np.full(len(F[m_index,mask]),(0.3-0.15*np.log10(mm))*(1-(logp[mask]-1)/(logpt-1)))
    return F
    '''
    F = 0.3-0.15*np.log10(m1)
    if m1<=6.5:
        logpt=8-m1
    else:
        logpt=1.5
    if logp<1:
        return F
    elif logp<logpt:
        F*=(1-(np.log10(p)-1)/(logpt-1))
        return F
    else:
        return 0
        '''
def normalize_xi(l,s,ftwin=0):
    l = np.array([l]).reshape(-1)
    s = np.array([s]).reshape(-1)
    ftwin = np.array([ftwin]).reshape(-1)
    
    xi2=np.zeros(l.shape)
    epsilon = 1e-10
    mask =  np.logical_and(np.abs(l+1)>epsilon,np.abs(s+1)>epsilon)
    gl=l[mask]
    gs=s[mask]
    #xi2[mask]=(gl+1)*(gs+1)/((gl-gs)*0.3**(gl+1)-0.3**(gl-gs)*0.1**(gs+1)*(gl+1)+gs+1)
    xi2[mask]=(0.3**(gl-gs)/(gs+1)*(0.3**(gs+1)-0.1**(gs+1))+(1-0.3**(gl+1))/(gl+1)/(1-ftwin[mask]))**(-1)
    mask =  np.logical_and(np.abs(l+1)>epsilon,np.abs(s+1)<epsilon)
    gl=l[mask]
    #xi2[mask] = (gl+1)/(1+(np.log(3)*gl+np.log(3)-1)*0.3**(gl+1))
    xi2[mask] = (0.3**(gl+1)*np.log(3)+(1-0.3**(gl+1))/(gl+1)/(1-ftwin[mask]))**(-1)
    mask =  np.logical_and(np.abs(l+1)<epsilon,np.abs(s+1)>epsilon)
    gs=s[mask]
    #xi2[mask] = (gs+1)*0.3**(gs+1)/(0.3**(gs+1)-0.1**(gs+1)-np.log(0.3)*(gs+1)*0.3**(gs+1))
    xi2[mask] = (0.3**(-1-gs)/(gs+1)*(0.3**(gs+1)-0.1**(gs+1))-np.log(0.3)/(1-ftwin[mask]))**(-1)
    
    mask =  np.logical_and(np.abs(l+1)<epsilon,np.abs(s+1)<epsilon)
    gs=s[mask]
    xi2[mask] = (np.log(3)+np.log(10/3)/(1-ftwin[mask]))**(-1)
    
    xi1=xi2*0.3**(l-s)
    return xi1,xi2
def final_joint(m1,q,p):
    q = np.array([q]).reshape(-1)
    p = np.array([p]).reshape(-1)
    m1 = np.array([m1]).reshape(-1)
    res = np.zeros((len(m1),len(q),len(p)))
    for m_index in range(len(m1)):
        mm=m1[m_index]
        for p_index in range(len(p)):
            pp = p[p_index]
            ppdf = P_distribution(mm,np.log10(pp))
            res[m_index,:,p_index] = ppdf*q_pdf(mm,q,np.log10(pp))

    return res
def q_pdf(m1,q,logp):
    q = np.array([q]).reshape(-1)
    res = np.zeros((len(q)))
    ftwin =  float(Ftwins(m1,10**logp))
    g_largeq = gamma_largeq(m1,10**logp)
    g_smallq = gamma_smallq(m1,10**logp)
    xi1,xi2=normalize_xi(g_largeq,g_smallq,ftwin)
    mask = np.logical_and(q>=0.1,q<0.3)
    res[mask] =  xi1*q[mask]**g_smallq
    mask = np.logical_and(q>=0.3,q<0.95)
    res[mask] = xi2*q[mask]**g_largeq
    
    mask = np.logical_and(q>=0.95,q<=1)
    res[mask] = xi2*q[mask]**g_largeq+20*ftwin/(1-ftwin)*I_largeq(xi2,g_largeq)
    return res.reshape(-1)
def I_largeq(xi2,gamma_largeq,qmin=0.3,qmax=1):
    gamma_largeq = np.array([gamma_largeq]).reshape(-1)
    xi2 = np.array([xi2]).reshape(-1)
    res = np.zeros(len(gamma_largeq))+xi2
    epsilon=1e-10
    mask = np.abs(gamma_largeq+1)<epsilon
    res[mask] *= np.log(qmax/qmin)
    res[~mask] *= (qmax**(gamma_largeq[~mask]+1)-qmin**(gamma_largeq[~mask]+1))/(gamma_largeq[~mask]+1)
    return res
def Inverse_q(m1,logp,y):
    y=np.array([y]).reshape(-1)
    q=np.zeros(len(y))
    g_smallq = gamma_smallq(m1,10**logp)
    g_largeq = gamma_largeq(m1,10**logp)
    ftwin = Ftwins(m1,10**logp).reshape(-1)
    xi1,xi2=normalize_xi(g_largeq,g_smallq,ftwin)
    mask =  np.logical_and(y>0,y<=I_smallq(xi1,g_smallq))
    if np.abs(g_smallq+1)<1e-10:
        q[mask]= 0.1*np.exp(y[mask]/xi1)
    else:
        q[mask] = ((g_smallq+1)*y[mask]/xi1+0.1**(g_smallq+1))**(1/(g_smallq+1))
        
    mask =  np.logical_and(y>I_smallq(xi1,g_smallq),y<=I_smallq(xi1,g_smallq)+I_largeq(xi2,g_largeq,0.3,0.95))
    if np.abs(g_largeq+1)<1e-10:
        q[mask] = 0.3*np.exp((y[mask]-I_smallq(xi1,g_smallq))/xi2)
    else:
        q[mask]=((y[mask]-I_smallq(xi1,g_smallq)+xi2/(g_largeq+1)*0.3**(g_largeq+1))*(g_largeq+1)/xi2)**(1/(g_largeq+1))

    mask =  np.logical_and(y>=I_smallq(xi1,g_smallq)+I_largeq(xi2,g_largeq,0.3,0.95),y<=1)
    mask1 = np.logical_and(mask,ftwin>0)
    q[mask1]=(y[mask1]-I_smallq(xi1,g_smallq)-I_largeq(xi2,g_largeq,0.3,0.95))/I_largeq(xi2,g_largeq)/ftwin*(1-ftwin)/20+0.95
    
    mask2 = np.logical_and(mask,ftwin==0)
    if np.abs(g_largeq+1)<1e-10:
        q[mask2] = 0.3*np.exp((y[mask2]-I_smallq(xi1,g_smallq))/xi2)
    else:
        q[mask2]=((y[mask2]-I_smallq(xi1,g_smallq)+xi2/(g_largeq+1)*0.3**(g_largeq+1))*(g_largeq+1)/xi2)**(1/(g_largeq+1))

    q[q>1]=np.random.random(len(q[q>1]))*0.05+0.95
    return q

def I_smallq(xi1,gamma_smallq,qmin=0.1,qmax=0.3):
    gamma_smallq = np.array([gamma_smallq]).reshape(-1)
    xi1 = np.array([xi1]).reshape(-1)
    res = np.zeros(len(gamma_smallq))+xi1
    epsilon = 1e-10
    mask = np.abs(gamma_smallq+1)<epsilon
    res[mask] *= np.log(qmax/qmin)
    res[~mask] *= (qmax**(gamma_smallq[~mask]+1)-qmin**(gamma_smallq[~mask]+1))/(gamma_smallq[~mask]+1)
    return res
def largeq_P_distribution(m1,logp):
    logp = np.array(logp).reshape(-1)
    fq3=np.zeros(logp.shape)
    alpha=0.018;dlogp=0.7;
    f1= 0.02+0.04*np.log10(m1)+0.07*(np.log10(m1))**2
    f27=0.039+0.07*np.log10(m1)+0.01*np.log10(m1)**2
    f55 = 0.078-0.05*np.log10(m1)+0.04*np.log10(m1)**2
    boundary = np.array([0.2,1,2.7-dlogp,2.7+dlogp,5.5,8])
    indices = np.digitize(logp, boundary).reshape(-1)
    func_list = [f1,
                 f1,
                 lambda logp:f1+(logp-1)/(1.7-dlogp)*(f27-f1-alpha*dlogp),
                 lambda logp:f27+alpha*(logp-2.7),
                 lambda logp:f27+alpha*dlogp+(logp-2.7-dlogp)/(2.8-dlogp)*(f55-f27-alpha*dlogp),
                 lambda logp:f55*np.exp(-0.3*(logp-5.5)),
                 f55*np.exp(-0.3*(8-5.5))
                 ]
    for i,index in enumerate(indices):
        fq3[i]=process_single_element(func_list[index],logp[i])
    return fq3
def P_distribution(m1,logp):
    logp = np.array([logp]).reshape(-1)
    fq3=largeq_P_distribution(m1,logp)
    g_smallq = gamma_smallq(m1,10**logp)
    g_largeq = gamma_largeq(m1,10**logp)
    ftwin = Ftwins(m1,10**logp).reshape(-1)
    xi1,xi2=normalize_xi(g_largeq,g_smallq,ftwin)
    fac=I_smallq(xi1,g_smallq)
    fq3 = fq3/(1-fac)/normalize_p_num(m1)
    #normal = normalize_p(m1)
    return fq3.reshape(-1)
def I_p(m1,logp):
    if logp<=1:
        return (1-0.3**(-0.6)/1.1*(0.3**1.1-0.1**1.1)*(0.3**(-0.6)/1.1*(0.3**1.1-0.1**1.1)+(0.7+0.15*np.log10(m1))/(1-0.3**0.5)))**(-1)*plargeq_cdf(m1,1)
    
    return 
def Binary_Fraction(m1):
    m1=np.array([m1]).reshape(-1)
    bf = np.zeros(len(m1))
    boundary = np.array([0.8,1.2,2,5,9,16])
    indices = np.digitize(m1, boundary).reshape(-1)
    func_list = [0.4,0.4,0.59,0.76,0.84,0.94,1]
    for i,index in enumerate(indices):
        bf[i]=process_single_element(func_list[index],m1[i])
    return bf

def q_cdf(m1,logp,q):
    q = np.array([q]).reshape(-1)
    g_smallq = gamma_smallq(m1,10**logp)
    g_largeq = gamma_largeq(m1,10**logp)
    ftwin = Ftwins(m1,10**logp).reshape(-1)
    xi1,xi2=normalize_xi(g_largeq,g_smallq,ftwin)
    CDF = np.zeros(len(q))
    mask =  np.logical_and(q>0.1,q<=0.3)
    for qindex in np.where(mask)[0]:
        qq = q[qindex]
        CDF[qindex] = I_smallq(xi1,g_smallq,0.1,qq)
    mask =  np.logical_and(q>0.3,q<=0.95)
    for qindex in np.where(mask)[0]:
        qq = q[qindex]
        CDF[qindex] = I_largeq(xi2,g_largeq,0.3,qq)+I_smallq(xi1,g_smallq)
    mask = np.logical_and(np.logical_and(q>0.95,q<=1),ftwin>0)
    CDF[mask] = I_smallq(xi1,g_smallq)+I_largeq(xi2,g_largeq,0.3,0.95)+I_largeq(xi2,g_largeq)*ftwin/(1-ftwin)*20*(q[mask]-0.95)
    
    mask = np.logical_and(np.logical_and(q>0.95,q<=1),ftwin==0)
    for qindex in np.where(mask)[0]:
        qq = q[qindex]
        CDF[qindex] = I_smallq(xi1,g_smallq)+I_largeq(xi2,g_largeq,0.3,qq)
    return CDF
def Binary_Function(m1):
    #return -0.55*np.exp(-0.2*m1)+1
    m1 = np.array([m1]).reshape(-1)
    logm=np.log10(m1)
    res = np.ones(len(m1))
    mask = m1<=30
    res[mask] = -0.188*logm[mask]**2+0.631*logm[mask]+0.476
    return res
def normalize_p(m1,alpha=0.018,dlogp=0.7):
    f1  = 0.02+0.04*np.log10(m1)+0.07*(np.log10(m1))**2
    f27 = 0.039+0.07*np.log10(m1)+0.01*np.log10(m1)**2
    f55 = 0.078-0.05*np.log10(m1)+0.04*np.log10(m1)**2
    k1 = (f27-f1-alpha*dlogp)/(1.7-dlogp)
    k2 = (f55-f27-alpha*dlogp)/(2.8-dlogp)
    k3 = f27+alpha*dlogp
    res = f1*(2.5-dlogp)+(dlogp**2/2-1.7*dlogp+1.445)*k1 +f27*(2.7+dlogp)+alpha*((2.7+dlogp)**2/2-2.7*(2.7+dlogp))-(f27*(2.7-dlogp)+alpha*((2.7-dlogp)**2/2-2.7*(2.7-dlogp)))+k3*5.5+(5.5**2/2-(2.7+dlogp)*5.5)*k2-(k3*(2.7+dlogp)-(2.7+dlogp)**2*k2/2)+(1-np.exp(-0.3*2.5))*f55/0.3
    return res
def npoly(x,a,b,c,d,e,f,g):
    return a*x**6+b*x**5+c*x**4+d*x**3+e*x**2+f*x+g
def normalize_p_num(m1):
    if m1<=1.2:
        ns = [0.03607359735052491,-0.25490907315343003,0.7717391384588859,-1.2977563974283755,1.299706224915645,-0.6709258188614733,0.6156848559315645]
    elif m1<=3.5:
        ns = [0.00014567582297694183,-0.0023292930922643694,0.015945349278521922,-0.056500919387135105,0.11994681871195678,-0.013079644659572263,0.42621831386759307]
    elif m1<=6:
        #return 0.09383158*m1**1.25320938+0.39990994
        ns = [-0.00013830125167401452,0.002838764497424726,-0.02154267640630644,0.07222741215769514,-0.08631508889838688,0.112649341574276,0.42033014286683046]
    else:
        #return 1.74227285*m1**0.20342096-1.308756
        ns = [ -9.774361294889914e-12,3.6480456747838973e-09,-5.569803003961848e-07,4.5401630774181484e-05,-0.002220260802053327,0.08050627352519167,0.8881471820764169]
    return npoly(m1,*ns)
def plargeq_cdf(m1,logp):
    logp = np.array(logp).reshape(-1)
    fq3=np.zeros(logp.shape)
    alpha=0.018;dlogp=0.7;
    f1= 0.02+0.04*np.log10(m1)+0.07*(np.log10(m1))**2
    f27=0.039+0.07*np.log10(m1)+0.01*np.log10(m1)**2
    f55 = 0.078-0.05*np.log10(m1)+0.04*np.log10(m1)**2
    k1 = (f27-f1-alpha*dlogp)/(1.7-dlogp)
    k2 = (f55-f27-alpha*dlogp)/(2.8-dlogp)
    k3 = f27+alpha*dlogp
    boundary = np.array([0.2,1,2.7-dlogp,2.7+dlogp,5.5,8])
    indices = np.digitize(logp, boundary).reshape(-1)
    func_list = [0,
                 lambda logp:f1*logp-f1*0.2,
                 lambda logp:f1*(logp-0.2)+(logp**2/2-logp+1/2)*k1,
                 lambda logp:f1*(2.5-dlogp)+(dlogp**2/2-1.7*dlogp+1.445)*k1  +f27*logp+alpha*(logp**2/2-2.7*logp)-(f27*(2.7-dlogp)+alpha*((2.7-dlogp)**2/2-2.7*(2.7-dlogp))),
                 lambda logp:f1*(2.5-dlogp)+(dlogp**2/2-1.7*dlogp+1.445)*k1 +f27*(2.7+dlogp)+alpha*((2.7+dlogp)**2/2-2.7*(2.7+dlogp))-(f27*(2.7-dlogp)+alpha*((2.7-dlogp)**2/2-2.7*(2.7-dlogp)))+k3*logp+(logp**2/2-(2.7+dlogp)*logp)*k2-(k3*(2.7+dlogp)-(2.7+dlogp)**2*k2/2),
                 lambda logp:f1*(2.5-dlogp)+(dlogp**2/2-1.7*dlogp+1.445)*k1 +f27*(2.7+dlogp)+alpha*((2.7+dlogp)**2/2-2.7*(2.7+dlogp))-(f27*(2.7-dlogp)+alpha*((2.7-dlogp)**2/2-2.7*(2.7-dlogp)))+k3*5.5+(5.5**2/2-(2.7+dlogp)*5.5)*k2-(k3*(2.7+dlogp)-(2.7+dlogp)**2*k2/2)+(1-np.exp(-0.3*(logp-5.5)))*f55/0.3,
                 f1*(2.5-dlogp)+(dlogp**2/2-1.7*dlogp+1.445)*k1 +f27*(2.7+dlogp)+alpha*((2.7+dlogp)**2/2-2.7*(2.7+dlogp))-(f27*(2.7-dlogp)+alpha*((2.7-dlogp)**2/2-2.7*(2.7-dlogp)))+k3*5.5+(5.5**2/2-(2.7+dlogp)*5.5)*k2-(k3*(2.7+dlogp)-(2.7+dlogp)**2*k2/2)+(1-np.exp(-0.3*(8-5.5)))*f55/0.3,
                 ]
    for i,index in enumerate(indices):
        fq3[i]=process_single_element(func_list[index],logp[i])
    
    g_largeq = gamma_largeq(m1,10**logp)
    g_smallq = gamma_smallq(m1,10**logp)
    ftwin = Ftwins(m1,10**logp).reshape(-1)
    xi1,xi2=normalize_xi(g_largeq,g_smallq,ftwin)
    #fq3 = fq3*(1+I_smallq(xi1,g_smallq)*(1-ftwin)/I_largeq(xi2,g_largeq))/Binary_Function(m1)
    fq3 = fq3*(1-ftwin)/I_largeq(xi2,g_largeq)/Binary_Function(m1)
    normal = process_single_element(func_list[-1],8)*(1-ftwin)/I_largeq(xi2,g_largeq)/Binary_Function(m1)
    return fq3/normal

def p_cdf(m1,logp):
    if isinstance(logp, (int, float)):
        return np.cumsum(P_distribution(m1,np.linspace(0.2,logp,1000))*(logp-0.2)/1000)[-1]
    else:
        logp = np.array([logp]).reshape(-1)
        return np.cumsum(P_distribution(m1,logp)*(logp[-1]-logp[0])/len(logp))
def Inverse_p(m1,y,nbins=1000):
    y=np.array([y]).reshape(-1)
    logp=np.linspace(0.2,8,nbins)
    pcdf = p_cdf(m1,logp)
    indices = np.digitize(y, pcdf).reshape(-1)
    res = np.zeros(len(y))
    for i,index in enumerate(indices):
        if index==nbins:
            res[i]=8
        else:
            res[i]=(logp[index-1]+logp[index])/2
    return res
def ChabrierIMF(M):
    A = 0.8524635550318918
    B = 0.23791194888102837
    mc =0.079
    sigma =0.69
    ind = (M <= 1)
    #phil =  A * np.exp(pow((np.log10(M) - np.log10(0.079)), 2.0) / (-2.0 * pow(0.69, 2))) / M
    phil = A/M*np.exp(-np.log10((M/mc))**2/2/sigma**2)
    phig = B*M**(-2.3)
    return phil*ind+(1-ind)*phig


def ChabrierIMF_by_Mass(M):
    A = 0.8524635550318918
    B = 0.23791194888102837
    mc =0.079
    sigma =0.69
    ind = (M <= 1)
    phil = M* A * np.exp(pow((np.log10(M) - np.log10(mc)), 2.0) / (-2.0 * pow(sigma, 2))) / M
    phig = M*B*M**(-2.3)
    return phil*ind+(1-ind)*phig

def sample_massive_star_con(dM,MaxMass = 100,GMC_MASSCUT = 8):
    lowMassFrac_by_mass = quad(ChabrierIMF_by_Mass, 0.1, GMC_MASSCUT)[0]
    lowMassAverageMass = quad(ChabrierIMF_by_Mass, 0.1, GMC_MASSCUT)[0]
    norm_by_number = quad(ChabrierIMF, 0.1, MaxMass)[0]
    lowMassFrac_by_number = quad(ChabrierIMF, 0.1, GMC_MASSCUT)[0]/quad(ChabrierIMF, 0.1, MaxMass)[0]
    CDF_intercept_by_mass = lowMassFrac_by_mass + 0.237912 / 0.3 * pow(GMC_MASSCUT,-0.3)
    CDF_intercept = (0.23791194888102837 / 1.3 * pow(GMC_MASSCUT,-1.3))/norm_by_number/(1-lowMassFrac_by_number) 
    if np.random.rand() > (1-lowMassFrac_by_number)*norm_by_number*dM:
        return -1
    else: 
        PsampleMassive = np.random.rand()
        M = pow(1/(-1.3*(1-lowMassFrac_by_number)*norm_by_number*(PsampleMassive-CDF_intercept)/0.23791194888102837),1/1.3)
        return M
def Companion_IMF(m2,IMF=ChabrierIMF,MaxMass = 100,GMC_MASSCUT = 8):
    return Binary_Function(m2)*quad(lambda m:IMF(m2)*quad(lambda logp:final_joint(m,m2/m,10**logp),0.2,8)[0],m2,MaxMass)[0]

def generate_binary(Num,MaxMass = 100,GMC_MASSCUT = 8):
    norm_by_number = quad(ChabrierIMF, 0.1, MaxMass)[0]
    lowMassFrac_by_number = quad(ChabrierIMF, 0.1, GMC_MASSCUT)[0]/quad(ChabrierIMF, 0.1, MaxMass)[0]
    CDF_intercept = (0.23791194888102837 / 1.3 * pow(GMC_MASSCUT,-1.3))/norm_by_number/(1-lowMassFrac_by_number) 
    PsampleMassive = np.random.rand(Num)
    Masses = pow(1/(-1.3*(1-lowMassFrac_by_number)*norm_by_number*(PsampleMassive-CDF_intercept)/0.23791194888102837),1/1.3)
    StarPeriod = np.zeros(Num)
    StarMR = np.zeros(Num)
    for i in range(Num):
        StarPeriod[i] = Inverse_p(Masses[i],np.random.uniform(0,1))
        StarMR[i] = Inverse_q(Masses[i],StarPeriod[i],np.random.uniform(0,1))
    return np.array([Masses,StarPeriod,StarMR]).T

#========================================= Binary Interaction Case========================================
import h5py
import numpy as np
import math

# Constants
GRAVITY = 6.67430e-11  # m^3 kg^-1 s^-2
M_PI = math.pi
SOLAR_RADIUS = 6.957e8  # meters

def Rochelobe_radius(logp, q, m1):
    q = 1/q
    m1 = m1 * All.UnitMass_in_g
    p = math.pow(10, logp) * 24 * 3600  # convert from days to seconds
    a = math.pow(GRAVITY * m1 * (1 + 1/q) * math.pow(p, 2) / (4 * M_PI * M_PI), 1/3)
    return (0.49 * math.pow(q, 2.0/3.0) / (0.69 * math.pow(q, 2.0/3.0) + math.log(1 + math.pow(q, 1.0/3.0)))) * a / SOLAR_RADIUS

def interp_3d(Table, i, iz_low, iz_high, im_low, im_high, ir_low, ir_high, dz, dm, dr):
    # Split the long interpolation formula into multiple lines for clarity
    result =(1 - dz) * ((1 - dm) * ((1 - dr) * Table[iz_low][im_low][ir_low][i] + dr * Table[iz_low][im_low][ir_high][i]) + dm * ((1 - dr) * Table[iz_low][im_high][ir_low][i] + dr * Table[iz_low][im_high][ir_high][i])) + dz * ((1 - dm) * ((1 - dr) * Table[iz_high][im_low][ir_low][i] + dr * Table[iz_high][im_low][ir_high][i]) + dm * ((1 - dr) * Table[iz_high][im_high][ir_low][i] + dr * Table[iz_high][im_high][ir_high][i]))
    return result
def get_z_indicies(value, array, offset=0):
    # Find indices for interpolation
    idx = np.searchsorted(array, value)
    low = max(0, idx - 1)
    high = min(len(array) - 1, idx)
    
    # Calculate delta
    if high == low:
        delta = 0.0
    else:
        delta = (value - array[low]) / (array[high] - array[low])
    
    return low, high, delta
def determine_case(items,UnitMass_in_g = 1.989e33,AccretionEffeciencyBeta = 0.5,Useisentropicenvelopemodel = 0):
    with h5py.File('Arepo_GFM_Tables/Yields/Qcrits.hdf5', 'r') as f:
        Qcrits = {
            'Metallicity': f['/Metallicities'][:],
            'logM': f['/logM'][:],
            'logRl': f['/logRL'][:],
            'Q_crit': f['/Qcrits'][:]
        }
                
        # Determine qadindex
        if AccretionEffeciencyBeta > 0.5:
            qadindex = 0
        elif AccretionEffeciencyBeta < 0.5:
            qadindex = 2
        else:
            qadindex = 1
        if Useisentropicenvelopemodel == 1:
            qadindex += 3
        # Get indices
        Casetag = np.zeros(len(items))
        for i,item in enumerate(items):
            m1, logp, q, metallicity = item
            Rl = Rochelobe_radius(logp, q, m1)
            iz_low, iz_high, dz = get_z_indicies(metallicity, Qcrits['Metallicity'], 1e-20)
            im_low, im_high, dm = get_z_indicies(math.log10(m1), Qcrits['logM'], 0)
            ir_low, ir_high, dr = get_z_indicies(math.log10(Rl), Qcrits['logRl'], -0.05)
                
            # Interpolate values
            qad = interp_3d(Qcrits['Q_crit'], qadindex, iz_low, iz_high, im_low, im_high, ir_low, ir_high, dz, dm, dr)
            qth = interp_3d(Qcrits['Q_crit'], 6, iz_low, iz_high, im_low, im_high, ir_low, ir_high, dz, dm, dr)
            qL2 = interp_3d(Qcrits['Q_crit'], 7, iz_low, iz_high, im_low, im_high, ir_low, ir_high, dz, dm, dr)
            
            # Debug print
            print(f"qad: {qad}, qth: {qth}, qL2: {qL2}")
            print(f"1/q: {1/q}")
            
            # Determine case
            if qad == -1 and qL2 == -1 and qth == -1:
                Casetag[i] = -1
            elif 1/q > qad:
                Casetag[i] = 1
            elif 1/q > qL2:
                Casetag[i] = 2
            elif 1/q > qth:
                Casetag[i] = 3
            else:
                Casetag[i] = 4
        return Casetag

def test_determine_case():
    # Get data ranges from Qcrits.hdf5
    with h5py.File('Arepo_GFM_Tables/Yields/Qcrits.hdf5', 'r') as f:
        logM_min = f['/logM'][0]
        logM_max = f['/logM'][-1]
        logRL_min = f['/logRL'][0]
        logRL_max = f['/logRL'][-1]
        print(f"logM range: [{logM_min}, {logM_max}]")
        print(f"logRL range: [{logRL_min}, {logRL_max}]")
    
    # Create test cases covering different scenarios
    test_cases = [
        # Test cases covering logM range
        (math.pow(10, logM_min), 0.0, 1.0, 0.02),  # Minimum mass
        (math.pow(10, logM_max), 0.0, 1.0, 0.02),  # Maximum mass
        
        # Test cases covering logRL range
        (1.0, 0.0, 1.0, 0.02),  # Default case
        (1.0, 0.0, 0.5, 0.02),  # Small q
        (1.0, 0.0, 2.0, 0.02),  # Large q
        
        # Test cases covering metallicity range
        (1.0, 0.0, 1.0, 0.01),  # Low metallicity
        (1.0, 0.0, 1.0, 0.03),  # High metallicity
        
        # Edge cases
        (0.5, -1.0, 0.3, 0.02),  # Small mass, small q
        (2.0, 1.0, 2.5, 0.02),   # Large mass, large q
    ]
    
    print("\nRunning test cases:")
    for i, (m1, logp, q, metallicity) in enumerate(test_cases):
        print(f"\nTest case {i+1}: m1={m1}, logp={logp}, q={q}, metallicity={metallicity}")
        result = determine_case(m1, logp, q, metallicity)
        print(f"Result: {result}")

if __name__ == '__main__':
    from scipy.integrate import quad
    import numpy as np
    from matplotlib import pyplot as plt
    from scipy.interpolate import interp1d
    normalize_func = interp1d(np.linspace(0.8,100,1000),np.loadtxt('/home/my/squareroot/arepo_z/assistlgh/arepo_processing/normalize_list.txt'))
    for i in range(100):
        m1=np.random.uniform(0.8,100)
        logp=np.linspace(0.2,7.9999999,100)
        val = quad(lambda x: P_distribution(m1,x),0.2,8)[0]
        print(val)