import numpy as np
def latexify_pict(address='',minicaption=False,quiet=False):
    '''
    address:list or str help to invert minipages to latex
    input:
        ['./00031403071/inring.png','hh']
    out:
        \begin{figure}[!h]
        \begin{minipage}{0.5\textwidth}
            \includegraphics[width=\textwidth]{./00031403071/inring.png}
            \caption{}
            \label{}
        \end{minipage}
        \begin{minipage}{0.5\textwidth}
            \includegraphics[width=\textwidth]{hh}
            \caption{}
            \label{}
        \end{minipage}
            \caption{}
            \label{}
        \end{figure}
    '''
    if type(address) is str:
        address=address.replace('\\','/')
        s='\\begin{figure}[!h]\n\\includegraphics[width=0.8\\textwidth]{'+address+'}\n\\caption{}\n\\label{}\n\\end{figure}\n'
    else:
        address=[i.replace('\\','/') for i in address]
        num=len(address)
        wid=1/num
        mini='\n\t\t\\caption{}\n\t\t\\label{}'if minicaption else ''
        minipage=lambda x:'\t\\begin{minipage}{'+'{:.1f}'.format(wid)+'\\textwidth}\n\t\t\\includegraphics[width=\\textwidth]{'+'{}'.format(x)+'}'+mini+'\n\t\\end{minipage}\n'
        s='\\begin{figure}[!h]\n'+''.join([minipage(i) for i in address])+'\\caption{}\n\\label{}\n\\end{figure}'
    if not quiet:
        print(s)
    return s

def tab_like(data,ty='',precision=8,*kwarg,**kwargs):
    '''
    data <ndarray,np.matrix>:2darray
    ty <str>:'tabular','matrix',''
    precision <int>:Reserved digits default 8.
    '''
    predata=np.array(data,dtype='U'+str(precision))
    pro_data=list(map('\t&\t'.join,predata))
    length=len(predata)
    if ty=='tabular':
        print('\\begin{table}[!h]')
        print('\t\\caption{}')
        print('\t\\label{}')
        print('\t\\begin{tabularx}{\\textwidth}{|X|'+'Y|'*(length-1)+'}')
        print('\t\t\\hline',end='\n\t\t')
        print('\t\\\\\\hline\n\t\t'.join(pro_data)+'\t\\\\\\hline')
        print('\t\\end{tabularx}')
        print('\\end{table}')
    elif ty=='matrix':
        print('$\\begin{pmatrix}')
        print('\\\\\n'.join(pro_data),end='\\\\\n')
        print('\\end{pmatrix}$')
    else:
        print('\n'.join(pro_data))