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

def mplstyle_rc():
    plt.style.use(pkg_resources.resource_filename(__name__, 'assistlgh.mplstyle'))
    plt.rcParams['axes.prop_cycle'] = cycler('color', ['#C64B2B','#5084C3','#E728F4','#f9ed69','#799936','#22BB9C','#C3E3E5','#F6A495','#907CC8','#6B3A7F'])
def auto_fontsize(figsize=4.8,fac=3):
    font_size = figsize*fac
    plt.rcParams.update({
        'font.size': font_size,
        'axes.titlesize': font_size,
        'xtick.labelsize': font_size,
        'ytick.labelsize': font_size,
    })
    
def more_cmap(color_bin = 5):
    from matplotlib.colors import LinearSegmentedColormap
    if color_bin ==2:
        colors = [(0,"#EE82EE"),(1,"#00D1FF")]
    elif color_bin==3:
        colors = [(0,'#1A2A6C'),(0.5,"#B21F1F"),(1,"#FDBB2D")]
    elif color_bin ==4:
        colors = [(0,"#0D375F"),(0.3155,"#BAD3E6"),(0.6565,"#F67579"),(1,"#C50020")]
    elif color_bin==5:
        colors = [(0,'#1F0654'),(0.2682,'#72197D'),(0.5063,'#AC1E1E'),(0.7651,'#E99227'),(1,'#FFEFBA')]

    return LinearSegmentedColormap.from_list('mycmap', colors)