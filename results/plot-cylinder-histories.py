#!/usr/bin/env python3
import argparse
import numpy as np
import scipy.integrate as integrate
import scipy.fftpack   as fftpack
import os.path
import json

import matplotlib
#matplotlib.use('pgf')
import matplotlib.pyplot as plt
from matplotlib import rc, rcParams


# Arguments
parser = argparse.ArgumentParser(description='results parser for high-fidelity CFD verification workshop: mesh motion test suite')
parser.add_argument("--save", action='store_true', default=False, help="Save time-histories to image files.")
args = parser.parse_args()



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


groups = {'UM':{'color':'y--',
                'config':{},
                'integrals':{'Y-Force':{},
                             'Work':{},
                             'Mass':{},
                             'Mass Error':{}
                             }
               },
          'AFRL':{'color':'b--',
                'config':{},
                'integrals':{'Y-Force':{},
                             'Work':{},
                             'Mass':{},
                             'Mass Error':{}
                             }
               }
         }
motions = ['M1','M2']





def load_participant_data(group,case,motion,h,p,t):
    """
    Look for participant data file. 
        If found return as numpy array.
        If NOT found, return False
    """
    
    # Load data
    file = f"{group}/{case}-{motion}-{h}-{p}-{t}.txt"
    if (os.path.isfile(file)):
        data = np.loadtxt(file, skiprows=1,delimiter=',',ndmin=2)
    else:
        print('Data not found: '+file)
        data = False

    return data




def process_data(data):
    """
    Take data that was read-in from a participant data file, process it, and 
    perform some rudimentary validation.

    Return individual time-histories, 
                      end-time-index (in-case last index is used for participant-provided integrated quantities), 
                      and a skip indicator (in-case start/end-time validation failed)
    """


    # Detect participant integrated quantities 
    participant_integrals = None
    if np.isnan(data[-1,0]):
        participant_integrals = {}
        participant_integrals['Y-Force']    = data[-1,1]
        participant_integrals['Work']       = data[-1,2]
        participant_integrals['Mass']       = data[-1,3]
        participant_integrals['Mass error'] = data[-1,4]
        data = np.delete(data,-1,0)



    # Pull out columns from data format
    time           = data[:,0]
    y_force        = data[:,1]
    work_integrand = data[:,2]
    mass           = data[:,3]
    mass_error     = data[:,4]


    # Validate start time
    skip=False
    if not np.isclose(time[0],0.):
        print(f"Start-time for data-set is not 0. Skipping...")
        skip=True

    # Validate stop time
    if motion == 'M1':
        if not np.isclose(time[-1],1.):
            print(f"End-time for data-set is not 1. Skipping...")
            skip=True
    elif motion == 'M2':
        if not np.isclose(time[-1],40.):
            print(f"End-time for data-set is not 40. Skipping...")
            skip=True

    return time,y_force,work_integrand,mass,mass_error,participant_integrals,skip



def get_max_h(config):
    """
    Take configuration dictionary read in from either Airfoil.json or Cylinder.json
    and detect max 'h'-index included in data-set.
    """
    max_h = None
    for h in reversed(hs):
        if h in config:
            max_h = h
            break

    if max_h is None:
        print("No 'h'-index descriptors detected in data set json configuration.")
        sys.exit()

    return max_h

def get_max_p(config):
    """
    Take configuration dictionary read in from either Airfoil.json or Cylinder.json
    and detect max 'p'-index included in data-set.
    """
    max_p = None
    for p in reversed(ps):
        if p in config:
            max_p = p
            break

    if max_p is None:
        print("No 'p'-index descriptors detected in data set json configuration.")
        sys.exit()

    return max_p

def get_max_t(group,geometry,motion):
    """
    Look at files in contributed data set for given <group,geometry> and
    detect the maximum 't'-index submitted.
    """
    # Return all files in submission directory
    files = os.listdir(f"{group}")

    # Check for ts entries in file name
    max_t = '0'
    for file in files:
        # Check if file include geometry and is a data file (.txt)
        if (geometry in file) and (motion in file) and ('.txt' in file):
            for t in reversed(ts):
                # Update max_t if higher t was found in file name
                if (t in file) and (t > max_t):
                    max_t = t
                    break

    return max_t


# Plot time-histories
for group in groups:
    for motion in motions:

        # Read data-set configuration
        f = open(f"{group}/Cylinder.json")
        groups[group]['config'] = json.load(f)

        # Get max resolution
        max_h = get_max_h(groups[group]['config'])
        max_p = get_max_p(groups[group]['config'])
        max_t = get_max_t(group,'Cylinder',motion)

        # Load max-resolution data
        data = load_participant_data(group,'Cylinder',motion,max_h,max_p,max_t)

        # Plot time-histories
        if isinstance(data, np.ndarray):
            time,y_force,work_integrand,mass,mass_error,participant_integrals,skip = process_data(data)

            if not skip: 

                groups[group]['integrals']['Y-Force'] = integrate.simps(y=y_force,        x=time)
                groups[group]['integrals']['Work']    = integrate.simps(y=work_integrand, x=time)
                groups[group]['integrals']['Mass']    = integrate.simps(y=mass,           x=time)

                color = groups[group]['color']
                if (motion == 'M1'):
                    ax_M1_1.plot(time,y_force,        color,linewidth=1.0,label=f"{group}: {max_h}-{max_p}-{max_t}")
                    ax_M1_2.plot(time,work_integrand, color,linewidth=1.0,label=f"{group}: {max_h}-{max_p}-{max_t}")
                    ax_M1_3.plot(time,mass,           color,linewidth=1.0,label=f"{group}: {max_h}-{max_p}-{max_t}")
                    ax_M1_4.plot(time,mass_error,     color,linewidth=1.0,label=f"{group}: {max_h}-{max_p}-{max_t}")

                    print(group)
                    print('Y-Force', groups[group]['integrals']['Y-Force'])
                    print('Work',    groups[group]['integrals']['Work'])
                    print('Mass',    groups[group]['integrals']['Mass'])
                elif (motion == 'M2'):
                    ax_M2_1.plot(time,y_force,        color,linewidth=1.0,label=f"{group}: {max_h}-{max_p}-{max_t}")
                    ax_M2_2.plot(time,work_integrand, color,linewidth=1.0,label=f"{group}: {max_h}-{max_p}-{max_t}")
                    ax_M2_3.plot(time,mass,           color,linewidth=1.0,label=f"{group}: {max_h}-{max_p}-{max_t}")
                    ax_M2_4.plot(time,mass_error,     color,linewidth=1.0,label=f"{group}: {max_h}-{max_p}-{max_t}")


            


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

if args.save:
    M1.savefig('Cylinder_M1_Histories.png', bbox_inches='tight', dpi=800)
    M2.savefig('Cylinder_M2_Histories.png', bbox_inches='tight', dpi=800)

plt.show()




