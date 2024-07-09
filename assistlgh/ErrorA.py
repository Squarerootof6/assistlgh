import numpy as np
import pandas as pd
from sympy import *
import seaborn as sns
from scipy.stats import pearsonr
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
import latexify
from math import *
from sklearn.metrics import r2_score
import random
import os 
import re
from matplotlib import gridspec
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

  
def rof(func,data):
    '''
    rof:result of functions
    
    func:name of function. the function must be designed to accept one single tuple,which contain all variance needed,
    all the variance are unpacked inside the function.
    _____________make sure two parameters'data are included in your s.______
    example:
            def E(s):
                m,c=s
                return m*(c**2)
    
    data: a 2-dim ndarray,data(first for 'm' and second for 'c') are packed in ordered
    example:
                data=np.array([(1,2,3,4,5),(1,2,3,4,5,6)])
    '''
    va=tuple(map(np.average,data))
    print('average:',va)
    st=list(map(np.std,data))
    print('std:',st)
    print('va=',va)
    value=func(va)
    print('value:',value)
    para=func.__code__.co_varnames
    X=symbols(','.join(para[1:]))
    

    print('paras are:',','.join(para[1:]))

    
    sigma=dict(zip(X,st))
    values=dict(zip(X,va))
    result=0
    print('传递系数and结果：')
    for x in X:
        a=diff(func(X),x)
        print('a{}'.format(x),'=',a)
        av=diff(func(X),x).subs(values)
        result+=(av*sigma[x])**2
    result=sqrt(result)
    print('result= ',value,' +- ',result)
    return value,result


##draw points
def draw(x,y,label='',marker='o',s=10,new=True,plot=False,text=True):
    '''
    draw a scatter for your data
    x,y,label,marker,s,c,new
'''
    c='C{}'.format(random.randint(0,8))
    plt.rcParams['font.sans-serif'] = ['SimHei']  
    plt.rcParams['axes.unicode_minus'] = False    
    sns.set(font='SimHei')   
    if new==True:
        fig,ax=plt.subplots(dpi=120)
    if plot==False:
        plt.scatter(x,y,c=c,label=label,lw=2,marker=marker,s=s)
    else:
        plt.plot(x,y,c=c,label=label,lw=2,marker=marker)
    plt.grid(ls='--',alpha=0.8)
    
    count=1
    if text==True:
        for c,d in zip(x,y):
            if count%2!=0:
                plt.text(c,d,'('+r'%.3g'%c+',%.3g'%d+')',ha='left',va='bottom',color='k',fontsize=7)
            count+=1
    plt.legend()
#set_function



def exp(x,a,b,c):
    return a*np.exp(b*x)+c
def poly2(x,a,b,c):
    return a*x**2+b*x+c
def linear(x,a,b):
    return a*x+b
def poly3(x,a,b,c,d):
    return a*x**3+b*x**2+c*x+d
#fit


def fit(func,x,y):
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
def save(fname):
    plt.savefig(fname)
    
def slope(popt,x,y):
    yb=popt[0]*x+popt[1]
    Q=np.sum((y-yb)**2)
    N=(len(y)-2)
    print('数据残余平方和：\t',Q)
    sigma2=Q/(N-2)
    print('数据残差平方和：\t',sigma2)
    sigmab2=N/(N*np.sum(x**2)+(np.sum(x))**2)*sigma2
    print('斜率方差：\t',sigmab2)
    print('斜率标准差：\t',sigmab2**(1/2))
    return 0
def walk_files(doubleline,keywords=0,types='excel'):
    '''
    
    Parameters
    ----------
    doubleline : str
        use '/' or '\\' to describe your address. 
        example:'D:\\python\\CA3+'
    keywords : str
        Identify the key words of the file
    types : str
        file type
        

    Returns pd.DataFrame
    -------
    TYPE
        DESCRIPTION.

    '''

    print('keyword is ',keywords)
    def judge(string,key=keywords):
        return key in string
    

    if keywords ==0:
        keywords=types
    address=list(os.walk(doubleline))
    if address == []:
        a=[doubleline]
        print('original address:',a[-1])
    else:
        address=list(address[0])
        print('original address:',address)
        address[-1]=list(filter(judge, address[-1]))
        print('filtered address:',address)
        def add(a,b=address[0]):
            return b+'\\'+a
        a=list(map(add,address[-1]))

    print('get address:\n',a)
    data=pd.DataFrame()
    for ad in a:
        df=eval('pd.read_'+types+'(ad)')
        data=pd.concat([data,df],axis=1,ignore_index=True)#横向拼接
        #data.rename(columns={-1:ad,-2:ad+'1'},inplace=True) 
    return data

#调试代码
'''def E(s):
    m,c=s
    return m*(c**2)
def A(s):
    d,l=s
    return (l**2)/(2*d*1000000)
data2=np.array([(0.28135,0.263275,0.2723125),(589.3,589.3,589.3)])
data=np.array([(1,2,3,4,5),(1,2,3,4,5,6)])

Error_synthesis(E,data)
Error_synthesis(A,data2)

v=np.array([0.5,0.8,1.2,1.5,2.0,3.9])
m=np.array([-0.2,-0.4,-0.6,-0.8,-1,-2])
fit('linear',m,v)
plt.show()
'''