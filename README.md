# PYEAD

Welcome to PYEAD!!

This is the PYthon Energy and Angle Distribution (PYEAD) calculator, a lightweight python tool to estimate energy and angle distributions of impinging ions through the sheath.

Note: This code is a python implementation and extension of an EAD code written by C. Chrobak and modified by S. Abe, named ieadkr3.pro.

Davis C. Easley (2022-)

## Description

This code is used for Monte Carlo estimations of the incident-ion EADs at a plasma-facing component (PFC) surface. Incident ions are distributed with a Thomson energy distribution at the sheath entrance and pushed by electromagnetic forces w/in the sheath.

The default sheath formulation is the Borodkina model (Borodkina, I., Borodin, D., Kirschner, A., Tsvetkov, I.V., Kurnaev, V.A., Komm, M., Dejarnac, R. and JET Contributors, (2016), An Analytical Expression for the Electric Field and Particle Tracing in Modelling of Be Erosion Experiments at the JET ITER-like Wall. Contrib. Plasma Phys., 56: 640-645. https://doi.org/10.1002/ctpp.201610032)

The code should be referenced with the following citations:

[1] D.C. Easley, et al., Dependence of high-Z redeposition on the field-to-surface pitch angle and other sheath parameters in tokamaks. Phys. Plasmas 1 May 2024; 31 (5): 052503. https://doi.org/10.1063/5.0187331

[2] D.C. Easley, et al., Predictive outlook for experiments resolving prompt vs local redeposition of high-Z materials in tokamaks.  2025 Plasma Phys. Control. Fusion 67 035023. https://doi.org/10.1088/1361-6587/adb64a


Further details on the physics of PYEAD can be found in:

D.C. Easley Ph.D. Dissertation (Section 3.3 and A.A):

[1] Easley, Davis C., "Prompt vs Local Redeposition: Model Refinement and Experimental Design for Understanding High-Z Net Erosion in Magnetic Confinement Fusion. " PhD diss., University of Tennessee, 2024. https://trace.tennessee.edu/utk_graddiss/10452

ieadkr3.pro script developed in:

[2] C.P. Chrobak et al 2018 Nucl. Fusion 58 106019. https://doi.org/10.1088/1741-4326/aad4c9

[3] S. Abe et al 2022 Nucl. Fusion 62 066001. https://doi.org/10.1088/1741-4326/ac3cdb

## Installing the code

To install, inside the top directory run:

```
python -m pip install -e .
```

To uninstall from anywhere:

```
python -m pip uninstall PYEAD
```

## Running the code

PYEAD is designed to be lightweight. To run an example, simply install as above, then:

```
from PYEAD import pyead
data = pyead()
data.calc_EAD()
```

## Code Details

This code uses calc_EAD as its primary workhorse to calculate the incident EAD of incoming ions. The inputs consist of the following defaults for a deuterium plasma case:

```
	#alphadeg: toroidal pitch angle, degrees from surface plane
	#	default alphadeg = 5
        #mii_amu: impurity or launched ion mass (amu)
	#	default mii_amu = mip_amu
        #mip_amu: background plasma ion mass (amu)
	#	default mip_amu = 2.016
        #mni: mach number of main ions
	#	default mni = 1.0
        #mnp: mach number of impurity ions
	#	default mnp = 1.0
        #Te=Ti (default): electron and ion temperature (eV)
	#	default Te = 25.0
        #B: magnetic field strength (T)
	#	default B = 2.0
        #Zip=Zii (default): plasma and impurity ion charge state
	# 	default Zip = 1.0
        #ne: background plasma elecron desnity (m^-3)
	#	default ne = 1.0e19
        #n_ions: number of test ions to track
	#	default n_ions = 500
        #lmps: sheath thickness = lmps*rho_i*cos(alpha)
	#	default lmps = 2.0
        #pal: number of parallel processes
	#	default pal = 2
```

These defaults may be changed directly as optional inputs to calc_EAD(...) or as part of an input deck.
If using an input deck, provide the path as an optional input to the function. The code will look for a file named "PYEAD_input" in the directory of the path given. If no path is given or the file cannot be found in the specificed path, the file will be sourced from the run directory. If no file is found in either place, the defaults will be assigned.
Any inputs to the function calc_EAD will overwrite inputs sourced from the input deck. 
The input deck should have the following form:

```
mip_amu = 2.016
alphadeg = 85.0
Te = 25.0
...

```

Any unassigned values will be set to the default parameters.


The order of operation in calc_EAD is the following:

1. Discretize the height above the PFC surface.

2. Calculate the electric field. The default is the Borodkina formulation, but other sheath calculations can be written up and used instead.

3. Initialize bins for initial and final energy and angles. Theta is measured from the surface normal and phi is measured from the magnetic field vector in the plane of the PFC. See Figure 1 in the first citation (Easley PoP 2024) for clarity. The initial gyro phase angle is randomized for each test ion and a Maxwellian distribution is used to assign the velocity, centered at the thermal sound speed.

4. In a parallelized Monte Carlo process, test particles are placed at a constant height above the PFC surface that is outside the sheath; then, positions and velocities are updated throughout the sheath travel using standard Newton and Runge-Kutta methods. Particles are pushed by the usual Lorentz force based on the sheath conditions. The timestep is updated to increase fidelity closer to the PFC surface.

5. The final energy and angles are binned when the particle connects to the PFC surface. If the particle does not connect, outputs are assigned to 0 or None as needed to maintain array sizes.


Results from calc_EAD can be output in two major file dumps:

```
data.write_plots()
data.write_out()
```

data.output_plots is used to plot a few standard distributions, and can be used with optional inputs:

```
data.output_plots(suppress=False,save=False)
```
to only view the plots without saving.

data.output_write() is used to write text file overviews of the results that are consistent with the outputs of ieadkr3.pro. 

Otherwise,

```
data.partsum 
```
contains the key quantities for each test particle. User postprocessing is encouraged since the output_() methods are only meant to provide quick overviews and not deep analytics. Most variables can be sourced directly as attributes of the pyead class for further analysis (e.g. data.theta, data.azim, data.finale).
