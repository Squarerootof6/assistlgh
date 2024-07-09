import numpy as np
from matplotlib import pyplot as plt
def mask_within(x,a,b):
    mask = np.logical_and(x>a,x<b)
    y=x[mask]
    return y,mask
def grad5(f,x,h=2e-7):
    if type(x)==str:
        return (-f(x+2*h)+8*f(x+h)-8*f(x-h)+f(x-2*h))/(12*h)
    elif type(x)==list:
        res = []
        for i in range(len(x)):
            y=np.array([x]*4).T
            y[i] += np.array([2*h,h,-h,-2*h])
            y=y.T
            res.extend((-f(y[0])+8*f(y[1])-8*f(y[2])+f(y[3]))/(12*h))
        return res

def fit_curve(func,x,y):
    '''
1.funname:'exp,poly2,poly3,linear' is included, please input str to use them.
//if you defined another function outside,the function name is expected.
2. x , y:only support single variable
example:

    ***def sss(theta,a,b,c):
    return ((a*np.cos(theta)/(b+c*np.sin(theta))))

    x=np.linspace(0,90,10)*np.pi/180
    y=[1.029,1.028,0.978,0.921,0.817,0.696,0.554,0.389,0.212,0.026]

    EA.draw(x,y)
    ***EA.fit('exp',x,y)
    ***EA.fit(sss,x,y)


'''
    x=np.array(x)
    y=np.array(y)
    if type(func)==str:
        func=eval(func)
    funname=func.__name__
    formula=str(latexify.with_latex(func))
    c=np.random.randint(0,8,1)
    try:
       popt,pcov=curve_fit(func,x,y)
    except funcerror:
        print(funcerror.arg)
        raise NameError('function not defined')
    popt,pcov=curve_fit(func,x,y)
    para=func.__code__.co_varnames
    print(para,formula)
    r2 = r2_score(y, func(x,*popt) )
    '''
    print(formula)
    for i,j in list(zip(para,popt)):
        print(i)
        print(j)
        formula=formula.replace(str(i),str(j))
    print(formula)
    '''
    if funname=='exp':
        plt.plot(x,func(x,*popt),ls='--',color='C{}'.format(c[0]),
                 label=r'${0}$ $fit:\hat{{y}} ={1:e}e^{{{2:,.3f}\hat{{x}}}}+{3:,.3f}$'.format(funname,popt[0],popt[1],popt[2]))
    elif funname == 'poly2':
        plt.plot(x,func(x,*popt),ls='--',color='C{}'.format(c[0]),
                 label=r'${0}$ $fit:\hat{{y}} ={1:e}\hat{{x}}^2{2:,.3f}\hat{{x}}+{3:,.3f}$\nR2_score={4:,.3f}'.format(funname,popt[0],popt[1],popt[2],r2))
    elif funname == 'poly3':
        plt.plot(x,func(x,*popt),ls='--',color='C{}'.format(c[0]),
                 label=r'${0}$ $fit:\hat{{y}} ={1:e}\hat{{x}}^3+{2:,.3f}\hat{{x}}^2+{3:,.3f}\hat{{x}}+{4:,.3f}$\nR2_score={}'.format(funname,popt[0],popt[1],popt[2],popt[3],r2))
    elif funname == 'linear':
        pcc=pearsonr(x,y)
        ci=int(input('ci?'))
        print('ci=',ci)
        sns.regplot(x,y,scatter=False,ci=ci,
                 label=r'${0}$ $fit:\hat{{y}} ={1:e}\hat{{x}}+{2:,.3f}$ r= {3:,.4f}'.format(funname,popt[0],popt[1],pcc[0]))
        slope(popt,x,y)
    else:
        X=np.linspace(x[0],x[-1],len(x)*5)
        plt.plot(X,func(X,*popt),ls='--',color='C{}'.format(c[0]),
                 label='${}$ fit:\n para={} \n R2_score={}'.format(formula,popt,r2))
    X=str(list(locals())[1])
    Y=str(list(locals())[2])
    plt.xlabel(X)
    plt.ylabel(Y)
    plt.title(Y+'-'+X+' Curve')
    plt.legend(loc='best',fontsize=8)
    print('the result[popt,pcov,r2]:\t',popt,pcov,r2)
    return popt,pcov,r2