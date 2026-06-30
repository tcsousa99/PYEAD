#############################################################################
#
# File name: PYEAD.py (the PYthon Energy and Angle Distribution estimator)
#
# Written by: D.C. Easley (ORNL: contact by email at easleydc@ornl.gov)
#
# Purpose: This program executes ion impact energy and angle
# distribution calculations as a Python conversion from existing IDL
# code, written by C. Chrobak and modified by S. Abe, named ieadkr3.pro.
# By converting to Python, this code may be more widely and easily used
# in the absence of an IDL license. To compensate for the longer
# computational runtime associated with open-source differential
# equation solvers, the monte carlo process has been isolated and
# parallelized (in contrast to the IDL version).
#
# Validated against: ieadrk3.pro updates through Abe 2022
#
#############################################################################

import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import time
from joblib import Parallel, delayed
from netCDF4 import Dataset

plt.rcParams.update({
        "font.size": 20,        # base font size
	"font.weight": 'bold',
        "axes.labelsize": 20,
        "axes.titlesize": 20,
	"axes.labelweight": 'bold',
	"axes.titleweight": 'bold',
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "legend.fontsize": 20,
})

class pyead():
	def __init__(self,source=True,path=None,alphadeg=None,mii_amu=None,mip_amu=None,mni=None,mnp=None,Te=None,Ti=None,B=None,Zip=None,Zii=None,ne=None,n_ions=None,out=None,name=None,forceds=None,noe=None,lmps=None,pal=None,ndvis=None):

		#The input values will be assigned in the following priority: 
		# Will check if source file is expected. If True, will read PYEAD_input input file for parameters. If False, will assign default values to all parameters except any values specified when creating the class instance.
		# Will check if path is specified (i.e. not None). If so, will look for PYEAD_input input file in that directory first. If not found, will attempt to use file in run directory.
		# Will overwrite any value specified in any manner above if assigned when calling the class instance.

		if source == True:
			if path is not None:
				inputfile = path+'/PYEAD_input'
				try:
					with open(inputfile,'r') as f: #Will try to find and read PYEAD_input in path
						print(f'PYEAD_input found in {path}. Using values as specified.')
					f.close()
				except:
					print(f'Error: PYEAD_input not found in {path}. Attempting to find in run directory...')
					inputfile = 'PYEAD_input'
			else:
				inputfile = 'PYEAD_input'
				try:
					with open(inputfile,'r') as f: #Will try to verify the existence of PYEAD_input in run directory.
						print(f'PYEAD_input found in run directory. Using values as specified.')
					f.close()
				except:
					print(f'Error: Unable to find PYEAD_input in run directory (missing path?).')

			try:
				with open(inputfile,'r') as f:
					for line in f:
						line = line.strip()
						if line and not line.startswith('#'):
							try:
								key,value = line.split('=',1)
								setattr(self,key.strip(),float(value.strip()))
								#print(f'key = {key} and value = {value}')
							except ValueError:
								print(f'Read error encountered at invalid line: {line}. Check PYEAD_input')
				f.close()

			except:
				print(f'Unable to read PYEAD_input. Setting missing values to default...')

		#Assign defaults iff necessary
		if alphadeg is not None:
			self.alphadeg = 90.0 - alphadeg #IMPORTANT NOTE: alphadeg input should be from surface, but mathematic formulation in this code treats alphadeg from surface normal. This reduces EAD noise accumulation from sin(small angle) in typical cases of interest where the Chodura sheath is large.
		elif not hasattr(self,'alphadeg'):
			self.alphadeg = 85.0 #Default B-to-PFC angle to 5 degrees from surface or 85 degrees from surface normal

		if B is not None:
			self.B = B
		elif not hasattr(self,'B'):
			self.B = 2.0 #Default toroidal magnetic field to 2.0 T

		if mip_amu is not None:
			self.mip_amu = mip_amu
		elif not hasattr(self,'mip_amu'):
			self.mip_amu = 2.016 #Default main plasma ion to D+

		if Zip is not None:
			self.Zip = Zip
		elif not hasattr(self,'Zip'):
			self.Zip = 1.0 #Default main plasma ion to D+

		if mnp is not None:
			self.mnp = mnp
		elif not hasattr(self,'mnp'):
			self.mnp = 1.0 #Default main plasma sound speed to Mach number 1.0

		if mii_amu is not None:
			self.mii_amu = mii_amu
		elif not hasattr(self,'mii_amu'):
			self.mii_amu = self.mip_amu #Default incident ion of interest to main plasma ion

		if Zii is not None:
			self.Zii = Zii
		elif not hasattr(self,'Zii'):
			self.Zii = self.Zip #Default incident ion of interest to main plasma ion charge state

		if mni is not None:
			self.mni = mni
		elif not hasattr(self,'mni'):
			self.mni = 1.0 #Default incident ion sound speed to Mach number 1.0

		if ne is not None:
			self.ne = ne
		elif not hasattr(self,'ne'):
			self.ne = 1.0e19 #Default electron density to 1x10^19 m^-3

		if Te is not None:
			self.Te = Te
		elif not hasattr(self,'Te'):
			self.Te = 25.0 #Default electron temperature to 25 eV

		if Ti is not None:
			self.Ti = Ti
		elif not hasattr(self,'Ti'):
			self.Ti = self.Te #Default ion temperature to electron temperature under isothermal assumption

		if lmps is not None:
			self.lmps = lmps
		elif not hasattr(self,'lmps'):
			self.lmps = 2.0 #Default to L_mps = 2.0*rho_i*sin(alpha)

		if forceds is not None:
			self.forceds = forceds
		elif not hasattr(self,'forceds'):
			self.forceds = 0.0 #Default does not force a manually assigned Debye sheath contribution

		if noe is not None:
			self.noe = noe
		elif not hasattr(self,'noe'):
			self.noe = False #Default does not turn electric field off

		if n_ions is not None:
			self.n_ions = n_ions
		elif not hasattr(self,'n_ions'):
			self.n_ions = 500 #Default to 500 test particles

		if pal is not None:
			self.pal = pal
		elif not hasattr(self,'pal'):
			self.pal = 2 #Default to 2 parallel processors

		if ndvis is not None:
			self.ndvis = ndvis
		elif not hasattr(self,'ndvis'):
			self.ndvis = 50000 #Set number of numerical divisions for sheath distance normal to surface

		#Verify precision
		self.mip_amu = np.double(self.mip_amu)
		self.mni = np.double(self.mni)
		self.mnp = np.double(self.mnp)
		self.alphadeg = np.double(self.alphadeg)
		self.Te = np.double(self.Te)
		self.B = np.double(self.B)
		self.Zip = np.double(self.Zip)
		self.Zii = np.double(self.Zii)
		self.ne = np.double(self.ne)
		self.mii_amu = np.double(self.mii_amu)
		self.Ti = np.double(self.Ti)
		self.forceds = np.double(self.forceds)
		self.lmps = np.double(self.lmps)

		self.n_ions = int(self.n_ions)
		self.pal = int(self.pal)
		self.ndvis = int(self.ndvis)

		#Set constants:
		self.e = np.double(1.602e-19) #Charge per electron in C
		self.Nav = np.double(6.022e23) #Atoms per mol
		self.eps = np.double(8.85e-12) #Vacuum permittivity in F/m
		self.me = np.double(9.109e-31) #Mass of electron in kg
		self.kb = self.e #Boltzmann's constant for T in eV

		#Convert units:
		self.alpha = self.alphadeg*(np.pi/180.0) #Degrees to radians
		self.mip = self.mip_amu/self.Nav/1000.0 #Plasma ion mass in kg
		self.mii = self.mii_amu/self.Nav/1000.0 #Impurity ion mass in kg
		self.qi = self.e*self.Zip #Plasma ion charge in C
		self.qii = self.e*self.Zii #Impurity ion charge in C

		#Calculate important quantities:
		self.By = np.double(-self.B*np.cos(self.alpha)) #y is along surface normal
		self.Bz = np.double(self.B*np.sin(self.alpha)) #z is along toroidal direction
		delta = np.double(1.0*np.pi/180.0) #toroidal field radial pitch angle
		self.Bx = np.double(-self.B*np.sin(delta)) #x is perpendicular to toroidal direction
		self.r_d = np.sqrt(self.eps*self.kb*self.Te/(self.ne*self.e**2.0)) #Debye length in m
		self.no = self.ne #Bulk electron density outside sheath
		self.csp = np.sqrt(self.kb*(self.Te+self.Ti)/self.mip) #Plasma sound speed from Stangeby 2000
		self.cip = np.sqrt(self.kb*self.Ti/self.mii) #Ion sound speed from Stangeby 2000
		self.cii = np.sqrt(2.0*self.kb*self.Ti/self.mii) #Thermal ion speed
		self.R_ics = self.mip*self.csp/(self.qi*self.B) #Larmor radius for plasma ion at plasma sound speed
		self.R_iimp = self.mii*self.cii/(self.qii*self.B) #Larmor radius for impurity ion at thermal ion speed
		self.rho_i = self.R_ics/self.r_d #Larmor radius with ion speed = sound speed
		self.L_mps = self.lmps*self.rho_i*np.sin(self.alpha) #magnetic pre-sheath length
		self.lamb_w = 0.5*np.log(2.0*np.pi*(self.me/self.mip)*((self.Te+self.Ti)/(self.Te))) #Borodkina eq.21 (Total potential drop through sheath)
		self.alpha_star = np.arccos(np.sqrt(2.0*np.pi*(self.me/self.mip)*(1.0+(self.Ti/self.Te))))*(180.0/np.pi) #Critical angle above which Debye sheath vanishes
		self.lamb_norm = self.kb*self.Te/self.e #Normalization constant for e-field strength lambda=e(phi-phi0)/(kb*Te)

		self.initialize_distances()

		self.name = 'output'

	def initialize_distances(self):
		self.Y = np.double(np.linspace(0.0,float(self.ndvis)-1.0,self.ndvis)/float(self.ndvis) * 0.010) #0.010 Y vector normal to surface, 10mm total length
		self.lamb = np.double(np.linspace(0.0,float(self.ndvis)-1.0,self.ndvis)) #Electric potential at the same
		self.xsi = self.Y/self.r_d #dimensionless distance
		print('Y and lamb reset')


	def initialize_EAD(self):
		#Define initial placeholders for calculated 3D angle distributions and energy distributions
		self.finale = np.zeros(self.n_ions,dtype=np.double) #Final energy of each test particle
		self.theta = np.zeros(self.n_ions,dtype=np.double) #Final theta of each test particle
		self.se_theta = np.zeros(self.n_ions,dtype=np.double) #Sheath-entrance theta of each particle
		self.azim = np.zeros(self.n_ions,dtype=np.double) #Final phi of each test particle
		self.se_azim = np.zeros(self.n_ions,dtype=np.double) #Sheath-entrance phi of each test particle

		self.finalebins = np.zeros(180,dtype=int) #Histogram of final energy
		self.se_ebins = np.zeros(180,dtype=int) #Histogram of sheath-entrance energy
		self.thetabins = np.zeros(180,dtype=int) #Histogram of final theta
		self.se_thetabins = np.zeros(180,dtype=int) #Histogram of sheath-entrance theta
		self.azimbins = np.zeros(180,dtype=int) #Histogram of final phi
		self.se_azimbins = np.zeros(180,dtype=int) #Histogram of sheath-entrance phi

		self.finalexthetabins = np.zeros((180,180),dtype=int) #2-D Histogram of final energy and theta
		self.finalexazimbins = np.zeros((180,180),dtype=int) #2-D Histogram of final energy and phi
		self.thetaxazimbins = np.zeros((180,180),dtype=int) #2-D Histogram of final theta and phi

		self.thetabinloc = np.double(np.linspace(0.0,180.0-1.0,180))
		self.azimbinloc = np.double(np.linspace(0.0,180.0-1.0,180))-90.0

		#Velocity bins
		self.vperpbins = np.zeros(180,dtype=int) #v_perpendicular histogram
		self.vparbins = np.zeros(180,dtype=int) #v_parallel histogram
		#self.vperpbins = (self.vperpbins.astype(np.double))/float(self.n_ions)
		#self.vparbins = (self.vparbins.astype(np.double))/float(self.n_ions)

		self.posbinloc = np.double(0.400*np.linspace(0.0,1000.0-1.0,1000)/1000.0 - 0.050) #xz distribution bin locations from -50 to 350 (mm)

		#Initialize xyz position bins
		self.finalxbins = np.zeros(1000,dtype=int)
		self.finalybins = np.zeros(1000,dtype=int)
		self.finalzbins = np.zeros(1000,dtype=int)
		#self.finalxbins = (self.finalxbins.astype(np.double))/float(self.n_ions)
		#self.finalybins = (self.finalybins.astype(np.double))/float(self.n_ions)
		#self.finalzbins = (self.finalzbins.astype(np.double))/float(self.n_ions)

		#Particle trajectory information structure
		partinfo = {'VPERP' : 0.0, 'VPAR' : 0.0, 'EI' : 0.0, 'EF' : 0.0, 'POLARI' : 0.0, 'POLARF' : 0.0, 'AZIMI' : 0.0, 'AZIMF' : 0.0, 'DELTAX' : 0.0, 'DELTAY' : 0.0, 'DELTAZ' : 0.0}
		self.partsum = [partinfo.copy() for k in range(self.n_ions)] #Set for each test particle

	def calc_EAD(self,sheath_model=None,verbose=True,write_plot=False,suppress_plt=True,write_out=False,file_name=None):

		#Calculates the EAD. This is the main function to call.

		if sheath_model is None  or  sheath_model=='Borodkina':
			self.calc_Borodkina_sheath()
		else:
			print(f'{sheath_model} is not supported yet. Please define function.')

		T_start = time.time()

		self.initialize_EAD()
		self.define_initial_velocities()

		if self.n_ions>200:
			result = Parallel(n_jobs=self.pal)(delayed(self.monte_carlo_EAD)(n,verbose) for n in range(self.n_ions))
			for n in range(self.n_ions):
				self.finale[n] = result[n][1]
				self.partsum[n]['EF'] = result[n][2]
				if result[n][3] is not None:
					self.finalebins[result[n][3]] += 1
				if result[n][4] is not None:
					self.se_ebins[result[n][4]] += 1 
				self.partsum[n]['DELTAX'] = result[n][5]
				self.partsum[n]['DELTAY'] = result[n][6]
				self.partsum[n]['DELTAZ'] = result[n][7]
				if result[n][8] is not None:
					self.finalxbins[result[n][8]] += 1
					self.finalybins[result[n][9]] += 1
					self.finalzbins[result[n][10]] += 1
				self.se_theta[n] = result[n][11]
				self.partsum[n]['POLARI'] = result[n][12]
				self.se_azim[n] = result[n][13]
				self.partsum[n]['AZIMI'] = result[n][14]
				if result[n][15] is not None:
					self.se_thetabins[result[n][15]] += 1
					self.se_azimbins[result[n][16]] += 1
				self.theta[n] = result[n][17]
				self.partsum[n]['POLARF'] = result[n][18]
				self.azim[n] = result[n][19]
				self.partsum[n]['AZIMF'] = result[n][20]
				if result[n][21] is not None:
					self.thetabins[result[n][21]] += 1
					self.azimbins[result[n][22]] += 1
				if result[n][23] is not None:
					self.vperpbins[result[n][23]] += 1
					self.vparbins[result[n][24]] += 1
				self.finalexthetabins[result[n][3]][result[n][21]] +=1
				self.finalexazimbins[result[n][3]][result[n][22]] +=1
				self.thetaxazimbins[result[n][22]][result[n][21]] +=1

		else:
			print('Insufficent n_ions to ensure accuracy. n_ions must be >200. \n')

		self.vperpbins = (self.vperpbins.astype(np.double))/float(self.n_ions)
		self.vparbins = (self.vparbins.astype(np.double))/float(self.n_ions)
		self.finalxbins = (self.finalxbins.astype(np.double))/float(self.n_ions)
		self.finalybins = (self.finalybins.astype(np.double))/float(self.n_ions)
		self.finalzbins = (self.finalzbins.astype(np.double))/float(self.n_ions)

		T_end = time.time()-T_start
		print('Time elapsed ='+str(T_end)+'\n')

		if write_plot == True:
			self.write_plot(name=file_name,suppress=suppress_plt,save=True)
			print('Plots saved')
		if write_out == True:
			self.write_out(name=file_name)
			print('Output saved')

		if verbose==True:
			print('mean theta= '+str(np.mean(self.theta))+'\n')
			print('stddev theta= '+str(np.std(self.theta))+'\n')
			print('mean azim= '+str(np.mean(self.azim))+'\n')
			print('stddev azim= '+str(np.std(self.azim))+'\n')
			print('mean finale= '+str(np.mean(self.finale))+'\n')
			print('stddev finale= '+str(np.std(self.finale))+'\n')

		print('EAD calculation complete')


	def calc_Borodkina_sheath(self):

		#Defines lamb and E_y for EAD calculation

		if (self.alphadeg > self.alpha_star) and (self.forceds == 0.0):
			print('alpha greater than critical angle ... no DS')
			self.lamb = self.lamb_w*np.exp(-2.0*self.xsi/self.L_mps) #electric potential across MPS only; Chrobak 2018 eq. A.2
			self.lambdebye = self.lamb*0.0
			self.lambmps = self.lamb
			self.xsi_mps = 0.0
		else:
			lamb_mps = np.log(np.cos(self.alpha)) #Chodura drop Borodkina eq. 6 / Stangeby eq. 2.109
			if self.forceds > 0.0:
				lamb_mps = (1.0-self.forceds)*self.lamb_w
			delta_mps = -4.0*np.log(np.cos(self.alpha))/(self.lmps*self.rho_i*np.sin(self.alpha))**2.0 #Borodkina eq. 12

			C1 = -delta_mps*lamb_mps-6.0*np.cos(self.alpha)
			a = (np.sqrt(-delta_mps*lamb_mps)-np.sqrt(2.0*np.exp(self.lamb_w)+4.0*np.cos(self.alpha)*np.sqrt(1.0-(self.lamb_w-lamb_mps))+C1))/(self.lamb_w-lamb_mps)
			Q = (1/a)*np.sqrt(2.0*np.exp(self.lamb_w)+4.0*np.cos(self.alpha)*np.sqrt(1.0-(self.lamb_w-lamb_mps))+C1)

			#Split the potential into two drops, Debye and MPS
			self.xsi_mps = -(1.0/a)*np.log((self.lamb_w-lamb_mps+Q)/Q)
			lamb_mps2 = self.lamb_w + Q - Q*np.exp(a*self.xsi_mps)
			i_debye = [i for i, j in enumerate(self.xsi) if j>=0.0 and j<=self.xsi_mps]
			i_mps = [i for i, j in enumerate(self.xsi) if j>=self.xsi_mps]

			#Debye layer solution
			self.lamb[i_debye] = self.lamb_w + Q - Q*np.exp(-a*self.xsi[i_debye])
			#MPS solution
			self.lamb[i_mps] = lamb_mps*np.exp(-np.sqrt(-(delta_mps/lamb_mps))*(self.xsi[i_mps]-self.xsi_mps))

			lambdebye = self.lamb*0.0
			lambdebye[i_debye] = self.lamb[i_debye]
			self.lambdebye = lambdebye
			lambmps = self.lamb*0.0
			lambmps[i_mps] = self.lamb[i_mps]
			self.lambmps = lambmps

		lamb_y = self.lamb*self.lamb_norm #Stop total electric potential in volts
		self.E_y = -1.0*self.der(self.Y,lamb_y) #E-field structure, SI units
		if self.noe != False:
			self.E_y = 0.0*lamb_y

	def define_initial_velocities(self):

		if not hasattr(self,'partsum'):
			print('Do not define initial velocities yet. Run initialize_EAD first!')

		else:
			#Determine gyro phase angle range of ion entering sheath
			#Specify a min and max initial gyro phase angle, then choose a set of random angles to use in that range
			#Refer to Schmid 2010 paper to calculate range of gyration phase angle for particles entering the sheath
			bigomega_min = np.double(0.0*np.pi/180.0)
			bigomega_max = np.double(360.0*np.pi/180.0)
			self.bigomega_rand = np.double(np.random.RandomState(61517).uniform(size=self.n_ions)*(bigomega_max-bigomega_min)+bigomega_min)

			#1D maxwellian distrubtion forward direction
			self.vparvec = np.sqrt((self.cii*np.double(np.random.RandomState(2017).normal(size=self.n_ions)))**2.0)

			#Clip to minimum of mni=0.01 to prevent backward-going ions
			clipind = [i for i, j in enumerate(self.vparvec) if j<0.01*self.csp]
			self.vparvec[clipind] = np.double(0.01*self.csp)

			#Maxwellian distribution of perpendicular ion velocities
			c = np.sqrt(self.kb*self.Ti/self.mii)
			self.vperpvec = np.sqrt((c*np.double(np.random.normal(size=self.n_ions)))**2.0 + (c*np.double(np.random.RandomState(1984).normal(size=self.n_ions)))**2.0)

			#Calculate velocity distribution histograms
			maxv = max([max(self.vperpvec),max(self.vparvec)])
			self.vbinloc = np.double(np.linspace(0.0,180.0-1.0,180))/180.0*maxv

			#Calculate total kinetic energy of ions entering sheath
			self.totale = 0.5*self.mii*(self.vparvec**2+self.vperpvec**2)/self.kb
			maxebin = max(self.totale)*3.0
			self.ebinloc = np.double(np.linspace(0.0,180.0-1.0,180))/180.0*maxebin

			#Fill in particle initial values:
			for i in range(len(self.partsum)):
				self.partsum[i]['VPERP'] = self.vperpvec[i]
				self.partsum[i]['VPAR'] = self.vparvec[i]
				self.partsum[i]['EI'] = self.totale[i]

	def monte_carlo_EAD(self,n,verbose=True):
		#Calculates the Energy and Angles of test particle n

		if verbose==True:
			#Show how far along you are...
			if n % math.floor(0.01*self.n_ions) == 0:
				print(str(n+1)+'/'+str(self.n_ions))
			else:
				print(str(n+1)+'/'+str(self.n_ions))


		v_perp = self.vperpvec[n]
		v_par = self.vparvec[n]
		perpind = np.argmin(abs(self.vbinloc-v_perp))
		parind = np.argmin(abs(self.vbinloc-v_par))

		if (self.mii_amu is not None) and (self.mii != self.mip):
			r_gyro = self.mii*v_perp/(self.qii*self.B)
		else:
			r_gyro = self.mip*v_perp/(self.qi*self.B)

		vx0 = -v_perp*np.cos(self.bigomega_rand[n]) #out of B-E plane
		vy0 = v_perp*np.sin(self.bigomega_rand[n])*np.sin(self.alpha)-v_par*np.cos(self.alpha) #in E-direction
		vz0 = v_par*np.sin(self.alpha)-v_perp*np.sin(self.bigomega_rand[n])*np.cos(self.alpha) #ion sound speed in B-direction

		#Generate variables and initialize
		dt = np.zeros(self.ndvis,dtype=np.double) #time step
		vx = np.zeros(self.ndvis,dtype=np.double) #x velocity
		vy = np.zeros(self.ndvis,dtype=np.double) #y velocity
		vz = np.zeros(self.ndvis,dtype=np.double) #z velocity
		ypos = np.zeros(self.ndvis,dtype=np.double) #y position
		xpos = np.zeros(self.ndvis,dtype=np.double) #x position
		zpos = np.zeros(self.ndvis,dtype=np.double) #z position
		speed = np.zeros(self.ndvis,dtype=np.double) #speed of particle
		ke = np.zeros(self.ndvis,dtype=np.double) #kinetic energy of particle
		tim = np.zeros(self.ndvis,dtype=np.double) #time

		vx[0] = vx0 #out of B-E plane
		vy[0] = vy0 #in E-direction
		vz[0] = vz0 #in B-direction

		#Start particle at position where electric field is 0.03 (1% of max normalized field of lambda=3)
		ypos[0] = (self.xsi[max([i for i, j in enumerate(self.lamb) if j<-0.03])])*self.r_d
		#Add one orbit diameter above this 1% level so that bottom of the orbit just dips into the mpsheath
		ypos[0] += 2.0*r_gyro*np.sin(self.alpha)

		speed[0] = np.sqrt(vx0**2.0+vy0**2+vz0**2)
		ke[0] = 0.5*self.mii*speed[0]**2

		#Track ion trajectories through layers of the sheath
		k=0
		dt = np.double(np.array([(0.01*2.0*np.double(np.pi)*r_gyro)/np.sqrt(vx0**2.0+vy0**2.0+vz0**2.0)]*self.ndvis))

		while (ypos[k] > 0.0) and (ypos[k] < 10.0*self.L_mps*self.r_d*3.0/self.lmps+2.0*r_gyro*np.sin(self.alpha)) and (k < self.ndvis-2):

			dt[k] = self.update_time(ypos=ypos[k],r_gyro=r_gyro,speed=speed[k],init_dt=dt[k]) #Update timestep

			# Find E-field at current position
			Eind = np.argmin(abs(self.Y-ypos[k]))
			Ey = 1.0*self.E_y[Eind]

			# Calculate next positions
			xpos[k+1] = self.NMx(xpos=xpos[k],vx=vx[k],vy=vy[k],vz=vz[k],dt=dt[k])
			ypos[k+1] = self.NMy(ypos=ypos[k],vx=vx[k],vy=vy[k],vz=vz[k],dt=dt[k],Ey=Ey)
			zpos[k+1] = self.NMz(zpos=zpos[k],vx=vx[k],vy=vy[k],vz=vz[k],dt=dt[k])

			# Calculate next velocities
			result = self.RK4(tim=tim[k],vx=vx[k],vy=vy[k],vz=vz[k],Ey=Ey,dt=dt[k])
			vx[k+1]=result.y[0][0]
			vy[k+1]=result.y[1][0]
			vz[k+1]=result.y[2][0]

			speed[k+1] = np.sqrt(vx[k+1]**2.0+vy[k+1]**2.0+vz[k+1]**2.0)
			ke[k+1] = 0.5*self.mii*(speed[k+1]**2.0)/self.e

			tim[k+1] = tim[k]+dt[k]

			k += 1

		if k < self.ndvis-2:
			k_flag = True
		else:
			k_flag = False

		out = self.write_particle(k_flag=k_flag,k=k,ke=ke,xpos=xpos,ypos=ypos,zpos=zpos,vx0=vx0,vy0=vy0,vz0=vz0,vx=vx,vy=vy,vz=vz,perpind=perpind,parind=parind,n=n)

		return out

	def der(self,X,x):
		#Numerical derivative consistent with ieadrk3.pro IDL code
		h = np.double(X[1]-X[0])
		N = len(x)
		y = np.double(np.zeros(N))
		y[0] = (-3*x[0]+4*x[1]-x[2])/(2*h)
		y[N-1] = (3*x[N-1]-4*x[N-2]+x[N-3])/(2*h)
		for i in range(N):
			if i!=0 and i!=N-1:
				y[i] = (x[i+1]-x[i-1])/(2*h)
		return np.double(y)

	def DVS(self,t,PV,Ey):
		#Equations of motion
		vx = PV[0]
		vy = PV[1]
		vz = PV[2]
		ax = np.double(self.qii / self.mii)*np.double(self.Bz*vy-self.By*vz)
		ay = np.double(self.qii / self.mii)*np.double(Ey-self.Bz*vx+self.Bx*vz)
		az = np.double(self.qii / self.mii)*np.double(vx*self.By-vy*self.Bx)
		return np.array([ax,ay,az])

	def update_time(self,ypos,r_gyro,speed,init_dt):
		dt = init_dt
		#Update timestep if position is within 2 gyro-orbits of surface
		if ypos < 2.0*r_gyro:
                        dt = 10.0*self.r_d/speed
		#If position is less than 5 Debye lengths from surface, timestep shrinks to time it takes to travel 0.25 Debye lengths
		if ypos < 5.0*self.r_d:
			dt = 0.25*self.r_d/speed
		return dt

	def NMx(self,xpos,vx,vy,vz,dt):
		#Newton method to update x-position
		x_new = xpos+vx*dt+0.5*((self.qii/self.mii)*(self.Bz*vy-self.By*vz))*dt**2.0
		return x_new

	def NMy(self,ypos,vx,vy,vz,dt,Ey):
		#Newton method to update y-position
		y_new = ypos+vy*dt+0.5*((self.qii/self.mii)*(Ey-self.Bz*vx+self.Bx*vz))*dt**2.0
		return y_new

	def NMz(self,zpos,vx,vy,vz,dt):
		#Newton method to update z-position
		z_new = zpos+vz*dt+0.5*((self.qii/self.mii)*(vx*self.By-vy*self.Bx))*dt**2.0
		return z_new

	def RK4(self,tim,vx,vy,vz,Ey,dt):
		#4th order Runge-Kutta method for calculating velocity
		H = dt
		t = tim
		pvx = vx
		pvy = vy
		pvz = vz
		PV = np.array([pvx,pvy,pvz])
		result = solve_ivp(lambda x, y: self.DVS(x,y,Ey),[t,t+2*H],PV,'RK45',t_eval=[t+H],first_step=H)
		return result

	def write_particle(self,k_flag,k,ke,xpos,ypos,zpos,vx0,vy0,vz0,vx,vy,vz,perpind,parind,n):

		if k_flag == True:

			#Update partsum with final values

			#Final Energy
			finale_n = ke[k]
			partsum_EF = finale_n

			#Fill in particle summary info structure

			ebinind = np.argmin(abs(self.ebinloc-finale_n)) #fill in energy distribution

			#Inital Energy
			se_ebinind = np.argmin(abs(self.ebinloc-self.totale[n])) #fill in initial energy distribution

			partsum_DELTAX = xpos[k]-xpos[0] #This is the radial direction
			partsum_DELTAY = ypos[k]-ypos[0] #This is the normal to surface direction
			partsum_DELTAZ = zpos[k]-zpos[0] #This is the toroidal field direction

			xbinind = np.argmin(abs(self.posbinloc-xpos[k]))
			ybinind = np.argmin(abs(self.posbinloc-ypos[k]))
			zbinind = np.argmin(abs(self.posbinloc-zpos[k]))

			#Sheath Incident Angle
			se_theta_n = 90.0 - np.double(np.arctan(-vy0/(np.sqrt(vz0**2.0+vx0**2.0))))*180.0/np.pi
			partsum_POLARI = se_theta_n

			se_azim_n = np.double(np.arctan(vx0/vz0))*180.0/np.pi
			partsum_AZIMI = se_azim_n

			se_thetabin = math.floor(se_theta_n)
			se_azimbin = 90+math.floor(se_azim_n)

			#Final angle
			theta_n = 90.0-np.double(np.arctan(-vy[k-1]/(np.sqrt(vz[k-1]**2.0+vx[k-1]**2.0))))*180.0/np.pi
			partsum_POLARF = theta_n
			azim_n = np.double(np.arctan(vx[k-1]/vz[k-1]))*180.0/np.pi
			partsum_AZIMF = azim_n

			thetabin = math.floor(theta_n)
			azimbin = 90+math.floor(azim_n)

		else:

			#Update partsum with 0's and None's

			finale_n = 0.
			partsum_EF = 0.
			ebinind = None
			se_ebinind = None
			partsum_DELTAX = 0.
			partsum_DELTAY = 0.
			partsum_DELTAZ = 0.
			xbinind = None
			ybinind = None
			zbinind = None
			se_theta_n = 0.
			partsum_POLARI = 0.
			se_azim_n = 0.
			partsum_AZIMI = 0.
			se_thetabin = None
			se_azimbin = None
			theta_n = 0.
			partsum_POLARF = 0.
			azim_n = 0.
			partsum_AZIMF = 0.
			thetabin = None
			azimbin = None
			perpind = None
			parind = None

		return n,finale_n,partsum_EF,ebinind,se_ebinind,partsum_DELTAX,partsum_DELTAY,partsum_DELTAZ,xbinind,ybinind,zbinind,se_theta_n,partsum_POLARI,se_azim_n,partsum_AZIMI,se_thetabin,se_azimbin,theta_n,partsum_POLARF,azim_n,partsum_AZIMF,thetabin,azimbin,perpind,parind,xpos[0],ypos[0],zpos[0],vx0,vy0,vz0

	def write_out(self,name=None):
		if name is None:
			name = 'output'

		self.out_commfile(name=name)
		self.out_distfile(name=name)
		self.out_mpsfile(name=name)
		self.out_posfile(name=name)
		self.out_partfile(name=name)

	def write_plot(self,cmap=None,bounds=None,suppress=True,save=False):
		if cmap is None:
			cmap = matplotlib.cm.plasma

		if bounds is None:
			bounds = np.linspace(0,1,11)

		self.plot_Efield(save=save)
		self.plot_1D_results(save=save)
		self.plot_2D_results(cmap=cmap,bounds=bounds,save=save)
		if suppress==False:
			plt.show()

	def out_commfile(self,name=None):
		if name is None:
			name = 'output'

		commfile = name+'_comments.txt'
		comlun = open(commfile,'w')
		comlun.write('// runid: '+name+'\n')
		comlun.write('// alphadeg_(deg), mii_(amu), mip_(amu), Te_(eV), B_(T), zi_plasma, zi_ion, ne_(m^-3), n_ions'+'\n')
		comlun.write(str(90.0-self.alphadeg)+", "+str(self.mii_amu)+", "+str(self.mip_amu)+", "+str(self.Te)+", "+str(self.B)+", "+str(self.Zip)+", "+str(self.Zii)+", "+str(self.ne)+", "+str(self.n_ions)+'\n')
		comlun.write('critical angle= '+str(90.0-self.alpha_star)+'\n')
		comlun.write('plasma thermal speed cip m/s= '+str(self.cip)+'\n')
		comlun.write('ion thermal speed cii m/s= '+str(self.cii)+'\n')
		comlun.write('plasma sound speed csp m/s= '+str(self.csp)+'\n')
		comlun.write('L_mps = '+str(self.L_mps*self.r_d)+'\n')
		comlun.write('r_d = '+str(self.r_d)+'\n')
		comlun.write('mean theta = '+str(np.mean(self.theta))+'\n')
		comlun.write('stddev theta = '+str(np.std(self.theta))+'\n')
		comlun.write('mean azim = '+str(np.mean(self.azim))+'\n')
		comlun.write('stddev azim = '+str(np.std(self.azim))+'\n')
		comlun.write('mean finale = '+str(np.mean(self.finale))+'\n')
		comlun.write('stddev finale = '+str(np.std(self.finale))+'\n')
		comlun.close()

	def out_distfile(self,name=None):
		if name is None:
			name = 'output'

		distfile = name+'_distributions.txt'
		dislun = open(distfile,'w')
		dislun.write('// runid: '+name+'\n')
		dislun.write('polar_deg, se_polardist, surf_polardist, azim_deg, se_azimdist, surf_azimdist, eV, se_eVdist, surf_eVdist, vbins, vperpdist, vpardist'+'\n')
		dnions = float(self.n_ions)
		for l in range(180):
			dislun.write(str(self.thetabinloc[l])+' '+'{:0.4e}'.format(self.se_thetabins[l]/dnions)+' '+'{:0.4e}'.format(self.thetabins[l]/dnions)+' '+str(self.azimbinloc[l])+' '+'{:0.4e}'.format(self.se_azimbins[l]/dnions)+' '+'{:0.4e}'.format(self.azimbins[l]/dnions)+' '+str(self.ebinloc[l])+' '+'{:0.4e}'.format(self.se_ebins[l])+' '+'{:0.4e}'.format(self.finalebins[l])+' '+str(self.vbinloc[l])+' '+'{:0.4e}'.format(self.vperpbins[l]/dnions)+' '+'{:0.4e}'.format(self.vparbins[l]/dnions)+'\n')
		dislun.close()

	def out_mpsfile(self,name=None):
		if name is None:
			name = 'output'

		mpsfile = name+'_mps.txt'
		mpslun = open(mpsfile, 'w')
		mpslun.write('y[m], y[m/rho_ion], y[m/r_d], E[V/m], lambda_norm, ne[E13cm-3], ne[norm], lambda_debye, lambda_mps'+'\n')
		mpslun.write('debye-mps-border(m) = '+str(self.xsi_mps*self.r_d)+'\n')
		for nd in range(int(self.ndvis/10)):
			nds = nd*10
			ne_d = np.exp(self.lamb)
			denE13cm3 = self.no*ne_d[nds]/1.0e19
			mpslun.write('{:0.3e}'.format(self.Y[nds])+',	'+'{:0.3e}'.format(self.Y[nds]/self.R_ics)+',	'+'{:0.3e}'.format(self.Y[nds]/self.r_d)+',	'+'{:0.3e}'.format(self.E_y[nds])+',	'+'{:0.3e}'.format(self.lamb[nds])+',	'+'{:0.3e}'.format(denE13cm3)+',	'+'{:0.3e}'.format(ne_d[nds])+',	'+'{:0.3e}'.format(self.lambdebye[nds])+',	'+'{:0.3e}'.format(self.lambmps[nds])+',	\n')
		mpslun.close()

	def out_posfile(self,name=None):
		if name is None:
			name = 'output'

		posfile = name+'_positiondist.txt'
		poslun = open(posfile, 'w')
		poslun.write('distance[m],deltax,deltay,deltaz'+'\n')
		for l in range(1000):
			poslun.write(str(self.posbinloc[l])+' '+'{:0.4e}'.format(self.finalxbins[l])+' '+'{:0.4e}'.format(self.finalybins[l])+' '+'{:0.4e}'.format(self.finalzbins[l])+'\n')
		poslun.close()

	def out_partfile(self,name=None):
		if name is None:
			name = 'output'

		partfile = name+'_particles.txt'
		partlun = open(partfile,'w')
		partlun.write('particle, vperp[m/s], vpar[m/s], par/perp, Ei[eV], Ef[eV], polari[deg], polarf[deg], azimi[deg], azimf[deg], deltax[m], deltay[m], deltaz[m] \n')
		for n in range(self.n_ions):
			partlun.write(str(n+1)+' '+'{:0.4e}'.format(self.partsum[n]['VPERP'])+' '+'{:0.4e}'.format(self.partsum[n]['VPAR'])+' '+str(self.partsum[n]['VPAR']/self.partsum[n]['VPERP'])+' '+'{:0.4e}'.format(self.partsum[n]['EI'])+' '+'{:0.4e}'.format(self.partsum[n]['EF'])+' '+str(self.partsum[n]['POLARI'])+' '+str(self.partsum[n]['POLARF'])+' '+str(self.partsum[n]['AZIMI'])+' '+str(self.partsum[n]['AZIMF'])+' '+str(self.partsum[n]['DELTAX'])+' '+str(self.partsum[n]['DELTAY'])+' '+str(self.partsum[n]['DELTAZ'])+'\n')
		partlun.close()

	def plot_Efield(self,lamb=None,name=None,suppress=True,save=False):
		if lamb is None:
			lamb = self.lamb

		ne_d = np.exp(lamb) #Density drop through the sheath

		plt.figure()
		plt.plot(self.Y/self.R_ics,lamb)
		plt.title('Normalized Potential')
		plt.xlabel('plasma ion gyro radii')
		plt.ylabel('electric potential')
		plt.tight_layout()
		if save == True:
			if name is None:
				name = self.name
			plt.savefig(f'{name}_SV.jpg')

		plt.figure()
		plt.plot(self.Y/self.R_ics,ne_d)
		plt.title('Normalized Density')
		plt.xlabel('plasma ion gyro radii')
		plt.ylabel('electron density')
		plt.tight_layout()
		if save == True:
			if name is None:
				name = self.name
			plt.savefig(f'{name}_Sn.jpg')

		if suppress==False:
			plt.show()

	def plot_theta_results(self,name=None,suppress=True,save=False):
		plt.figure()
		plt.plot(self.thetabinloc,self.thetabins/max(self.thetabins),label='Final')
		plt.xlim(0.0,180.0)
		plt.title('Polar inclination angle theta')
		plt.plot(self.thetabinloc,self.se_thetabins/max(self.se_thetabins),label='Initial')
		print('mean theta= '+str(np.mean(self.theta))+'\n')
		print('stddev theta= '+str(np.std(self.theta))+'\n')
		plt.legend()
		plt.xlabel('Polar (deg)')
		plt.ylabel('normalized counts')
		plt.tight_layout()
		if save == True:
			if name is None:
				name = self.name
			plt.savefig(f'{name}_P.jpg')

		if suppress==False:
			plt.show()

	def plot_azim_results(self,name=None,suppress=True,save=False):
		plt.figure()
		plt.plot(self.azimbinloc,self.azimbins/max(self.azimbins),label='Final')
		plt.xlim(-90.0,90.0)
		plt.title('Azimuthal ExB angle phi')
		plt.plot(self.azimbinloc,self.se_azimbins/max(self.se_azimbins),label='Initial')
		print('mean azim= '+str(np.mean(self.azim))+'\n')
		print('stddev azim= '+str(np.std(self.azim))+'\n')
		plt.legend()
		plt.xlabel('Azim (deg)')
		plt.ylabel('normalized counts')
		plt.tight_layout()
		if save == True:
			if name is None:
				name = self.name
			plt.savefig(f'{name}_A.jpg')

		if suppress==False:
			plt.show()

	def plot_energy_results(self,name=None,suppress=True,save=False):
		plt.figure()
		plt.plot(self.ebinloc,self.finalebins/max(self.finalebins),label='Final')
		plt.title('Impact energy distribution')
		plt.plot(self.ebinloc,self.se_ebins/max(self.se_ebins),label='Initial')
		plt.legend()
		plt.xlabel('Energy (eV)')
		plt.ylabel('normalized counts')
		print('mean finale= '+str(np.mean(self.finale))+'\n')
		print('stddev finale= '+str(np.std(self.finale))+'\n')
		plt.tight_layout()
		if save == True:
			if name is None:
				name = self.name
			plt.savefig(f'{name}_E.jpg')

		if suppress==False:
			plt.show()

	def plot_vel_results(self,name=None,suppress=True,save=False):
		plt.figure()
		plt.plot(self.vbinloc,(self.vperpbins+0.0001)/max(self.vperpbins),label='Perp')
		plt.title('Velocity distributions')
		plt.xlabel('velocity (m/s)')
		plt.ylabel('normalized counts')
		plt.ylim(0.001,1.0)
		plt.plot(self.vbinloc,(self.vparbins+0.0001)/max(self.vparbins),label='Par')
		plt.legend()
		plt.tight_layout()
		if save == True:
			if name is None:
				name = self.name
			plt.savefig(f'{name}_v.jpg')

		if suppress==False:
			plt.show()

	def plot_energy_vs_theta_results(self,cmap=None,bounds=None,name=None,suppress=True,save=False):

		if cmap is None:
			cmap = matplotlib.cm.plasma

		if bounds is None:
			bounds = np.linspace(0,1,11)

		norm = matplotlib.colors.BoundaryNorm(bounds,cmap.N,extend='neither')

		plt.figure()
		ax = plt.gca()
		Thetabin,Ebin = np.meshgrid(self.thetabinloc,self.ebinloc)
		plt.contourf(Thetabin,Ebin,self.finalexthetabins/(self.finalexthetabins.max()))
		plt.title('Energy vs Polar distribution')
		plt.xlabel('Polar (deg)')
		plt.ylabel('Energy (eV)')
		plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),ax=ax,label='normalized counts')
		plt.xlim(0.0,90.0)
		plt.tight_layout()
		if save == True:
			if name is None:
				name = self.name
			plt.savefig(f'{name}_EvP.jpg')

		if suppress==False:
			plt.show()

	def plot_energy_vs_azim_results(self,cmap=None,bounds=None,name=None,suppress=True,save=False):

		if cmap is None:
			cmap = matplotlib.cm.plasma

		if bounds is None:
			bounds = np.linspace(0,1,11)

		norm = matplotlib.colors.BoundaryNorm(bounds,cmap.N,extend='neither')


		plt.figure()
		ax = plt.gca()
		Azimbin,Ebin = np.meshgrid(self.azimbinloc,self.ebinloc)
		plt.contourf(Azimbin,Ebin,self.finalexazimbins/(self.finalexazimbins.max()))
		plt.title('Energy vs Azim distribution')
		plt.xlabel('Azim (deg)')
		plt.ylabel('Energy (eV)')
		plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),ax=ax,label='normalized counts')
		plt.xlim(-90.0,90.0)
		plt.tight_layout()
		if save == True:
			if name is None:
				name = self.name
			plt.savefig(f'{name}_EvA.jpg')

		if suppress==False:
			plt.show()


	def plot_azim_vs_energy_results(self,cmap=None,bounds=None,name=None,suppress=True,save=False):

		if cmap is None:
			cmap = matplotlib.cm.plasma

		if bounds is None:
			bounds = np.linspace(0,1,11)

		norm = matplotlib.colors.BoundaryNorm(bounds,cmap.N,extend='neither')

		plt.figure()
		ax = plt.gca()
		Thetabin,Azimbin = np.meshgrid(self.thetabinloc,self.azimbinloc)
		plt.contourf(Thetabin,Azimbin,self.thetaxazimbins/(self.thetaxazimbins.max()))
		plt.title('Azim vs Polar distribution')
		plt.xlabel('Polar (deg)')
		plt.ylabel('Azim (deg)')
		plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),ax=ax,label='normalized counts')
		plt.xlim(0.0,90.0)
		plt.ylim(-90.0,90.0)
		plt.tight_layout()
		if save == True:
			if name is None:
				name=self.name
			plt.savefig(f'{name}_AvP.jpg')

		if suppress==False:
			plt.show()

	def plot_1D_results(self,name=None,suppress=True,save=False):
		self.plot_theta_results(name=name,save=save)
		self.plot_azim_results(name=name,save=save)
		self.plot_energy_results(name=name,save=save)
		self.plot_vel_results(name=name,save=save)
		if suppress==False:
			plt.show()

	def plot_2D_results(self,cmap=None,bounds=None,name=None,suppress=True,save=False):

		if cmap is None:
			cmap = matplotlib.cm.plasma

		if bounds is None:
			bounds = np.linspace(0,1,11)


		self.plot_energy_vs_theta_results(cmap=cmap,bounds=bounds,name=name,save=save)
		self.plot_energy_vs_azim_results(cmap=cmap,bounds=bounds,name=name,save=save)
		self.plot_azim_vs_energy_results(cmap=cmap,bounds=bounds,name=name,save=save)
		if suppress==False:
			plt.show()
