import numpy as np
import astropy.units as u
import astropy.constants as c
import pandas as pd
import re
import inspect
GRAVITY = 6.6738e-8
class AllPara():
    def __init__(self,para_dir):
        f = open(para_dir)
        self.data = []
        for item in f.readlines():
            if item == '\n':
                continue
            a = item.split(re.search(r' +',item).group())
            a[-1] = a[-1].split('\n')[0].split(' ')[0]
            self.data.append(a)
        f.close()
        self.data = np.array(self.data)
        #self.data = pd.DataFrame(self.data[:,1],index=self.data[:,0]).T
        for param in self.data:
            param_name = param[0]
            param_value = param[1]
            try:
                setattr(self,param_name,float(param_value))
            except:
                continue
        self.UnitEnergy_in_cgs = (self.UnitVelocity_in_cm_per_s)**2
        self.UnitTime_in_s = self.UnitLength_in_cm/self.UnitVelocity_in_cm_per_s
        self.data = dict(self.data)
        if self.GravityConstantInternal == 0:
            self.G = GRAVITY / pow(self.UnitLength_in_cm, 3) * self.UnitMass_in_g * pow(self.UnitTime_in_s, 2);
        else:
            self.G = self.GravityConstantInternal
    def __repr__(self):
        return self.data
    def read_units(self):
        units_m = np.float64(self.data['UnitMass_in_g'])*u.g
        units_l = np.float64(self.data['UnitLength_in_cm'])*u.cm
        units_v = np.float64(self.data['UnitVelocity_in_cm_per_s'])*u.cm/u.s
        units_t = units_l/units_v
        units_e = (units_v**2).cgs
        return units_m,units_l,units_v,units_t,units_e
def Gas_Status(hdfdir,filename):
    with h5py.File(hdfdir+filename) as f:
        BoxSize = f['Header'].attrs['BoxSize']
        gas = f['PartType0']
        para = AllPara(hdfdir+'parameters-usedvalues')
        units_m,units_l,units_v,units_t,units_e = para.read_units()
        temperature = ((gas['InternalEnergy'])*units_e*2/3*c.m_p/c.k_B).to(u.K)
        density = (gas['Density']*units_m/units_l**3).to(u.M_sun/u.pc**3)
        ndensity = (gas['Density']*units_m/units_l**3/c.m_p).to(u.cm**(-3))
        pressure = (ndensity*c.k_B*temperature).to(u.g/u.cm/u.s**2)
        dis = np.linalg.norm(gas['Coordinates'][:]-BoxSize/2,axis=1)
        Vr = np.sum(((gas['Coordinates'][:].T-BoxSize/2)/dis).T*gas['Velocities'][:],axis=1)
        HII = 1-gas['HI_Fraction'][:]
        pos = gas['Coordinates'][:]
        return density,temperature,pressure,Vr,HII,pos

if __name__ == '__main__':
    from GMC_star_formation import gmc_starformation_criteria
    import h5py
    with h5py.File('/home/my/squareroot/arepo_z/run/largerun/MPGC/test/snapshot_027.hdf5','r') as f:
        para = AllPara('/home/my/squareroot/arepo_z/run/largerun/MPGC/test/parameters-usedvalues')
        NumPart0 = f['Header'].attrs['NumPart_Total'][0]
        gas = f['PartType0']
        for i in range(NumPart0):
            flag = gmc_starformation_criteria(i,gas,para)
            if flag==2:
                print(flag)