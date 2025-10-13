from matplotlib import pyplot as plt
from cycler import cycler
import time
def progress_bar(now,length,notes='',frequency=1):
    '''
    now:current progress
    
    length:whole mission
    '''
    if int(now/(length)*100) % frequency == 0 :
        string = ' '*10+' '*len(notes)+'                \r'
        print(string,end='')
        print('{:*<10s} {:d}% {} {}\r'.format('>'*int(now/(length)*10),int(now/(length)*100),notes,time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())),end='')


import pkg_resources
def load_mplstyle_path():
    return pkg_resources.resource_filename(__name__, 'assistlgh.mplstyle')

def mplstyle_rc(customap=False):
    plt.style.use(pkg_resources.resource_filename(__name__, 'assistlgh.mplstyle'))
    if customap:
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

def Polygon_plot(df,x,y,piece,vmin=-1,vmax=1,cmap= 'RdYlBu',**kwargs):
    from matplotlib.colors import ListedColormap        
    import numpy as np
    from matplotlib import pyplot as plt
    from matplotlib.cm import ScalarMappable
    import matplotlib.patches as patches

    fig,ax=plt.subplots(figsize=(8,3))
                         
    smB = ScalarMappable(cmap = ListedColormap(plt.get_cmap(cmap)(np.linspace(0,1,256))))
    smB.set_clim(vmin,vmax)
    p_list = np.unique(x)
    m_list = np.unique(y)
    q_list = np.unique(piece)
    
    defaultkw = {'numVertices':6,'radius':0.5,'orientation':np.deg2rad(90),'facecolor':'none','edgecolor':'w'}
    for key in defaultkw.keys():
        if key not in kwargs.keys():
            kwargs[key] = defaultkw[key]
    for index in len(data):
        p = x[index]
        q = piece[index]
        m = y[index]
        data = df[index]
        p_index = np.where(p_list==p)[0][0]
        m_index = np.where(m_list==m)[0][0]
        q_index = np.where(q_list==q)[0][0]
        center_x, center_y = p_index, m_index
        hexagon = patches.RegularPolygon(
            xy=[center_x , center_y],**kwargs)
        verts = hexagon.get_verts()[::-1]
        x1 = [center_x,center_y]
        boundary = np.linspace(0.1,1,7)
        i = np.digitize(q, boundary).reshape(-1)[0]-1
        x2=verts[i]
        x3=verts[i+1]
        facecolor = smB.to_rgba(data)
        triangle = patches.Polygon([x1, x2, x3], closed=True,edgecolor='white',facecolor=facecolor)
        ax.add_patch(triangle)
    ax.set_ylabel(r'Mass[$M_{\odot}$]')
    ax.set_ylim(-1,len(m_list));
    ax.set_yticks(np.arange(0,len(m_list),1));
    ax.set_yticklabels(np.int32(m_list));
    ax.set_xlabel('P(days)')
    ax.set_xticks(np.arange(0,len(p_list),1));
    ax.set_xlim(-1,len(p_list))
    ax.set_xticklabels(np.round(p_list,3),rotation=45,);
    ax.grid(alpha=0.5)
    cax = fig.add_axes([0.04,1,ax.get_position().width*1.24,0.03])
    cbar2 = fig.colorbar(smB, cax=cax,orientation='horizontal',ticklocation='top')
    return fig,ax