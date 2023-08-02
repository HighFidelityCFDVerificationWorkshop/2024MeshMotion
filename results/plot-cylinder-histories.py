#!/usr/bin/env python3
import numpy as np
import scipy.integrate as integrate
import scipy.fftpack   as fftpack
import os.path

import matplotlib
#matplotlib.use('pgf')
import matplotlib.pyplot as plt
from matplotlib import rc, rcParams


# remove or set to True (default) to trigger exception
rc_xelatex = {'pgf.rcfonts': False} 
matplotlib.rcParams.update(rc_xelatex)

rc('text', usetex=True)
rc('font', family='serif')
rcParams.update({'figure.autolayout': True})


M1 = plt.figure(figsize=(16,4))
M1.set_facecolor('White')
ax_M1_1 = M1.add_subplot(141)
ax_M1_2 = M1.add_subplot(142)
ax_M1_3 = M1.add_subplot(143)
ax_M1_4 = M1.add_subplot(144)

M2 = plt.figure(figsize=(16,4))
M2.set_facecolor('White')
ax_M2_1 = M2.add_subplot(141)
ax_M2_2 = M2.add_subplot(142)
ax_M2_3 = M2.add_subplot(143)
ax_M2_4 = M2.add_subplot(144)




motions = ['M1','M2','M3','M4']
hs = ['h0','h1','h2','h3','h4','h5',]
ps = ['p0','p1','p2','p3','p4','p5','p6']
ts = ['t0','t1','t2','t3','t4','t5']


groups = ['UM']
motions = ['M1','M2']



for group in groups:
    for motion in motions:
        skip=False

        #max_h = get_max_h(group,'Cylinder',motion)
        #max_p = get_max_p(group,'Cylinder',motion)
        #max_t = get_max_t(group,'Cylinder',motion)
        max_h = 'h3'
        max_p = 'p4'
        max_t = 't4'
        file = f"{group}/Cylinder-{motion}-{max_h}-{max_p}-{max_t}.txt"
    

        # Load data
        if (os.path.isfile(file)):
            data = np.loadtxt(file, skiprows=1,delimiter=',')
        else:
            print('Data not found: '+file)
            data = False


        if isinstance(data, np.ndarray):

            time           = data[:,0]
            y_force        = data[:,1]
            work_integrand = data[:,2]
            mass           = data[:,3]
            mass_error     = data[:,4]

            # Detect participant integrated quantities 
            end_time_index = -1
            if np.isnan(time[-1]):
                end_time_index = -2

            # Validate start time
            if not np.isclose(time[0],0.):
                print(f"Start-time for data-set {file} is not 0. Skipping...")
                skip=True

            # Validate stop time
            if motion == 'M1':
                if not np.isclose(time[end_time_index],1.):
                    print(f"End-time for data-set {file} is not 1. Skipping...")
                    skip=True
            elif motion == 'M2':
                if not np.isclose(time[end_time_index],40.):
                    print(f"End-time for data-set {file} is not 40. Skipping...")
                    skip=True

            
            nx      = len(time)
            xend    = 2.
            afrl_dx = xend/(nx-1)
            afrl_x  = np.linspace(0.,xend,nx)

            color = 'b--'
            
            if (motion == 'M1'):
                ax_M1_1.plot(time,y_force,        color,linewidth=1.0,label=group)
                ax_M1_2.plot(time,work_integrand, color,linewidth=1.0,label=group)
                ax_M1_3.plot(time,mass,           color,linewidth=1.0,label=group)
                ax_M1_4.plot(time,mass_error,     color,linewidth=1.0,label=group)
            elif (motion == 'M2'):
                ax_M2_1.plot(time,y_force,        color,linewidth=1.0,label=group)
                ax_M2_2.plot(time,work_integrand, color,linewidth=1.0,label=group)
                ax_M2_3.plot(time,mass,           color,linewidth=1.0,label=group)
                ax_M2_4.plot(time,mass_error,     color,linewidth=1.0,label=group)





ax_M1_1.legend()
ax_M1_2.legend()
ax_M1_3.legend()
ax_M1_4.legend()

ax_M2_1.legend()
ax_M2_2.legend()
ax_M2_3.legend()
ax_M2_4.legend()

ax_M1_1.set_xlabel('Time')
ax_M1_2.set_xlabel('Time')
ax_M1_3.set_xlabel('Time')
ax_M1_4.set_xlabel('Time')

ax_M2_1.set_xlabel('Time')
ax_M2_2.set_xlabel('Time')
ax_M2_3.set_xlabel('Time')
ax_M2_4.set_xlabel('Time')

ax_M1_1.set_ylabel('Force-Y')
ax_M1_2.set_ylabel('Work integrand')
ax_M1_3.set_ylabel('Mass')
ax_M1_4.set_ylabel('Mass error')

ax_M2_1.set_ylabel('Force-Y')
ax_M2_2.set_ylabel('Work integrand')
ax_M2_3.set_ylabel('Mass')
ax_M2_4.set_ylabel('Mass error')

ax_M1_1.set_xlim((0.,1.))
ax_M1_2.set_xlim((0.,1.))
ax_M1_3.set_xlim((0.,1.))
ax_M1_4.set_xlim((0.,1.))
 
ax_M2_1.set_xlim((0.,40.))
ax_M2_2.set_xlim((0.,40.))
ax_M2_3.set_xlim((0.,40.))
ax_M2_4.set_xlim((0.,40.))

M1.savefig('Cylinder_M1_Histories.png', bbox_inches='tight', dpi=800)
M2.savefig('Cylinder_M2_Histories.png', bbox_inches='tight', dpi=800)

plt.show()




