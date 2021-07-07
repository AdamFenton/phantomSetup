import matplotlib.pyplot as plt
import numpy as np
import phantomsetup
import os
import time
import sys
cwd = os.getcwd()
working_dir = cwd
utime = 5.022E6 # PHANTOM code unit for time in seconds
yr = 3.15E7 # Seconds in year
msun = 1.989E33
AU = 1.496E13

##############
# Disc setup #
##############
number_of_particles = 100_000
norbits = 5

radius_min = 10.0
radius_max = 100.0
disc_mass = 0.3

alpha_art = 0.1 # Artifical viscosity alpha
beta_art = 2.0  # Artifical viscosity beta (keep this as 2 to prevent interparticle penetration)

p_index = 2.05
R0 = 10
reference_radius = 10.0
###########################
# Equation of state setup #
###########################
ieos = 8
isink=1
q_index = 0.25 # sound speed profile exponent


my_temp_exp = 0.5 # Temperature profile exponent. This is always double q_index
T0 = 10
Tinf = 10


R0_temp = 0.25



##############
# Sink setup #
##############
qfacdisc = 0.25
stellar_mass = 1.0
stellar_accretion_radius = 1
stellar_position = (0.0, 0.0, 0.0)
stellar_velocity = (0.0, 0.0, 0.0)






################################################################################
# Calculate the simualtion run time based on number of orbits and size of disc #
################################################################################
period = (radius_max **(3/2)) # Outer orbital period in years.
tmax = ((period * yr * norbits)/utime) # Simulation runtime in code units to follow disc for norbits
time_between_dumps = tmax/100 # Time in code units between dumps, by default set to produce 100 dump files
physical_time = (tmax * utime)/yr
igas = phantomsetup.defaults.PARTICLE_TYPE['igas']
prefix = 'disc'
particle_type = igas

# print(phantomsetup.units.unit_string_to_cgs('au'))

length_unit = phantomsetup.units.unit_string_to_cgs('au')
mass_unit = phantomsetup.units.unit_string_to_cgs('solarm')
time_unit = phantomsetup.units.unit_string_to_cgs('code_units')

gravitational_constant = 1.0



def density_distribution(radius, p_index, disc_mass,R0,radius_min,radius_max):
    """Disc surface density distribution.
    """
    return phantomsetup.disc.my_surface_density(radius, p_index, disc_mass,R0,radius_min,radius_max)





setup = phantomsetup.Setup()

setup.prefix = prefix
setup.set_units(
    length=length_unit, mass=mass_unit, gravitational_constant_is_unity=True
)

tree_accuracy=0.3
rhocrit1 = 1e-13
gamma1 = 1.4
setup.set_run_option('tmax', tmax)
setup.set_run_option('dtmax', time_between_dumps)
setup.set_run_option('beta', beta_art)
setup.set_compile_option('GRAVITY', True)
setup.set_run_option('tree_accuracy', 0.3)
setup.set_run_option('tree_accuracy', tree_accuracy)
setup.set_run_option('rhocrit1', rhocrit1)
setup.set_run_option('gamma1', gamma1)

aspect_ratio = phantomsetup.eos.get_aspect_ratio_new(
T0, q_index,reference_radius,stellar_mass,gravitational_constant)


polyk = phantomsetup.eos.polyk_for_locally_isothermal_disc(
    q_index, reference_radius, stellar_mass, gravitational_constant,aspect_ratio)
print(polyk)
setup.set_equation_of_state(ieos=ieos, polyk=polyk)
setup.set_run_option('isink', isink)


setup.set_dissipation(disc_viscosity=True, alpha=alpha_art)



setup.add_sink(
    mass=stellar_mass,
    accretion_radius=stellar_accretion_radius,
    position=stellar_position,
    velocity=stellar_velocity,

)



disc = phantomsetup.Disc(
    particle_type=particle_type,
    T0 = T0,
    Tinf = Tinf,
    R0 = R0,
    R0_temp = R0_temp,
    number_of_particles=number_of_particles,
    disc_mass=disc_mass,
    density_distribution=density_distribution,
    radius_max=radius_max,
    radius_range=(radius_min, radius_max),
    q_index=q_index,
    qfacdisc = qfacdisc,
    my_temp_exp = my_temp_exp,
    p_index=p_index,
    aspect_ratio=aspect_ratio,
    reference_radius=reference_radius,
    rhocrit1 = rhocrit1,
    gamma1 = gamma1,
    stellar_mass=stellar_mass,
    gravitational_constant=gravitational_constant,
    # extra_args=(radius_critical,gamma),
    extra_args=(p_index,disc_mass,R0,radius_min,radius_max),
    # extra_args=(p_index,reference_radius)
)

setup.add_container(disc)


setup.write_dump_file(directory=working_dir)
setup.write_in_file(directory=working_dir)

#######################
#  Write README file  #
#######################
read_me_file = open('%s/disc_setup_README.txt' % cwd,'w')
print("#" * 50,file=read_me_file)
print("               Accretion disc setup               ",file=read_me_file)
print("-" * 50,file=read_me_file)
print('Stellar mass             = ', stellar_mass, 'solar masses',file=read_me_file)
print('Stellar accretion radius = ', stellar_accretion_radius, 'AU',file=read_me_file)
print('Disc mass                = ', disc_mass, 'solar masses',file=read_me_file)
print('Disc radius              = ', radius_max, 'AU',file=read_me_file)
print('Initial temperature T0   = ', T0, 'Kelvin',file=read_me_file)
print('Polyk                    = ', polyk, file=read_me_file)
print('Aspect ratio             = ', aspect_ratio, file = read_me_file)
print('Number of particles      = ', number_of_particles,file=read_me_file)
print('Number of orbits         = ', norbits,file=read_me_file)
print('Artifical viscosity α    = ', alpha_art,file=read_me_file)
print("-" * 50,file=read_me_file)
print(" Edit the disc.in file to change runtime options, ",file=read_me_file)
print("    any values not defined are set to default     ",file=read_me_file)
print("-" * 50,file=read_me_file)
print("  Disc evolution will be followed for %d yrs" % physical_time, file = read_me_file)
print("#" * 50,file=read_me_file)

# completion_message = open('/Users/adamfenton/Documents/PhD/yearTwo/ICGeneration/completion_message.txt','r')
# keep_the_change = open('/Users/adamfenton/Documents/PhD/yearTwo/ICGeneration/keep_the_change.txt','r')
#
# lines = [line.rstrip('\n') for line in completion_message]
# for line in lines:
#     print(line)
#     time.sleep(0.1)
# time.sleep(0.5)
# lines = [line.rstrip('\n') for line in keep_the_change]
# for line in lines:
#     print(line)
#     time.sleep(0.1)
