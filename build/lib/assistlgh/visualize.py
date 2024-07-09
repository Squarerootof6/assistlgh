from matplotlib import pyplot as plt
from cycler import cycler
def progress_bar(now,length,notes='',frequency=1):
    '''
    now:current progress
    
    length:whole mission
    '''
    if int(now/(length)*100) % frequency == 0 :
        string = ' '*10+' '*len(notes)+'                \r'
        print(string,end='')
        print('{:*<10s} {:d}% {}\r'.format('>'*int(now/(length)*10),int(now/(length)*100),notes),end='')


import pkg_resources
def load_mplstyle_path():
    return pkg_resources.resource_filename(__name__, 'assistlgh.mplstyle')

def mplstyle_rc(fig=None,fac=1):
    plt.style.use(pkg_resources.resource_filename(__name__, 'assistlgh.mplstyle'))
    plt.rcParams['axes.prop_cycle'] = cycler('color', ['#C64B2B','#F6A495','#E728F4','#CEB426','#799936','#22BB9C','#C3E3E5','#5084C3','#907CC8','#6B3A7F'])
    if fig is not None:
        fig_width, fig_height = fig.get_size_inches()
        font_size = min(fig_width, fig_height)*fac
        plt.rcParams.update({
            'font.size': font_size,
            'axes.titlesize': font_size,
            'axes.labelsize': font_size,
            'xtick.labelsize': font_size,
            'ytick.labelsize': font_size,
        })