import matplotlib.pyplot as plt
import numpy as np
import phantomsetup
import os
"This is a test"
"this is another test of time check"
cwd = os.getcwd()

utime = 5.022E6 # PHANTOM code unit for time in seconds
yr = 3.15E7 # Seconds in year
msun = 1.989E33
AU = 1.496E13

##############
# Disc setup #
##############
number_of_particles = 100_000
norbits = 5

radius_min = 1.0
radius_max = 100.0
disc_mass = 0.35

alpha_art = 1.0
beta_art = 2.0

p_index = 2.05


###########################
# Equation of state setup #
###########################
ieos = 3
isink=1
q_index = 0.25
qfacdisc = 0.5
T0 = 240
Tinf = 10
R0 = 10
R0_temp = 0.25
reference_radius = 1.0


##############
# Sink setup #
##############
stellar_mass = 1.0
stellar_accretion_radius = 1
stellar_position = (0.0, 0.0, 0.0)
stellar_velocity = (0.0, 0.0, 0.0)





# sigma_0 = ((((2-p_index)/(2*np.pi*(R0*AU)**2))*(disc_mass*msun))/((((R0*AU)**2+(radius_max*AU)**2)/((R0*AU)**2))**(1-(p_index/2))-(((R0*AU)**2+(radius_min*AU)**2)/((R0*AU)**2))**(1-(p_index/2))))
# print(sigma_0)

################################################################################
# Calculate the simualtion run time based on number of orbits and size of disc #
################################################################################
period = (radius_max **(3/2)) # Outer orbital period in years.
tmax = ((period * yr * norbits)/utime) # Simulation runtime in code units to follow disc for norbits
time_between_dumps = tmax/100 # Time in code units between dumps, by default set to produce 100 dump files

igas = phantomsetup.defaults.PARTICLE_TYPE['igas']
prefix = 'disc'
particle_type = igas


length_unit = phantomsetup.units.unit_string_to_cgs('au')
mass_unit = phantomsetup.units.unit_string_to_cgs('solarm')
gravitational_constant = 1.0


# def density_distribution(radius, radius_critical, gamma):
#     """Self-similar disc surface density distribution.
#
#     This is the Lyden-Bell and Pringle (1974) solution, i.e. a power law
#     with an exponential taper.
#     """
#     return phantomsetup.disc.self_similar_accretion_disc(radius, radius_critical, gamma) * sigma_crit
#

def density_distribution(radius, p_index, disc_mass,R0,radius_min,radius_max):
    """Disc surface density distribution.
    """
    return phantomsetup.disc.my_surface_density(radius, p_index, disc_mass,R0,radius_min,radius_max)


# def density_distribution(radius, p_index, reference_radius):
#     """Disc surface density distribution.
#     """
#     return phantomsetup.disc.power_law(radius, p_index, reference_radius)



setup = phantomsetup.Setup()
setup.prefix = prefix
setup.set_units(
    length=length_unit, mass=mass_unit, gravitational_constant_is_unity=True
)
tree_accuracy=0.3
setup.set_run_option('tmax', tmax)
setup.set_run_option('dtmax', time_between_dumps)
setup.set_run_option('beta', beta_art)
setup.set_compile_option('GRAVITY', True)
setup.set_run_option('tree_accuracy', 0.3)
setup.set_run_option('tree_accuracy', tree_accuracy)

aspect_ratio = phantomsetup.eos.get_aspect_ratio(
T0, q_index,reference_radius,stellar_mass,gravitational_constant)

# polyk = phantomsetup.eos.polyk_for_locally_isothermal_disc(
#     q_index, reference_radius,aspect_ratio, stellar_mass, gravitational_constant
# )
polyk = phantomsetup.eos.polyk_for_locally_isothermal_disc_mine(
    T0,q_index, reference_radius, stellar_mass, gravitational_constant
)
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
    p_index=p_index,
    aspect_ratio=aspect_ratio,
    reference_radius=reference_radius,
    stellar_mass=stellar_mass,
    gravitational_constant=gravitational_constant,
    # extra_args=(radius_critical,gamma),
    extra_args=(p_index,disc_mass,R0,radius_min,radius_max),
    # extra_args=(p_index,reference_radius)
)

setup.add_container(disc)





working_dir = cwd



setup.write_dump_file(directory=working_dir)
setup.write_in_file(directory=working_dir)



# print("#" * 50)
# print("               Accretion disc setup               ")
# print("-" * 50)
# print('Stellar mass             = ', stellar_mass, 'solar masses')
# print('Stellar accretion radius = ', stellar_accretion_radius, 'AU')
# print('Disc mass                = ', disc_mass, 'solar masses')
# print('Disc radius              = ', radius_max, 'AU')
# print('Number of particles      = ', number_of_particles)
# print('Number of orbits         = ', norbits)
# print("-" * 50)
# print(" Edit the disc.in file to change runtime options, ")
# print("    any values not defined are set to default     ")
# print("-" * 50)
# print("           Set the following in disc.in           ")
# print("-" * 50)
# print('tmax                     = ', tmax)
# print('dtmax                    = ', time_between_dumps)
# print("#" * 50)

# import matplotlib.pyplot as plt
# import numpy as np
# import phantomsetup
#
# import os
# import pickle
# cwd = os.getcwd()
#
#
# runtime_options = phantomsetup.defaults._RUN_OPTIONS # Read in the runtime option dict so the default values can be changed.
# defaults_header = phantomsetup.defaults.HEADER # Read in the runtime option dict so the default values can be changed.
#
#
#
# utime = 5.022E6 # PHANTOM code unit for time in seconds
# yr = 3.15E7 # Seconds in year
# msun = 1.989E33
# AU = 1.496E13
#
# ##############
# # Disc setup #
# ##############
# number_of_particles = 10_000
# norbits = 5
# radius_min = 1.0
# radius_max = 100.0
# alpha_artificial = 1.0
#
# p_index = 2.5
# disc_mass = 0.2
# radius_critical = 100
# gamma = 3/2
#
#
# sigma_crit = (((2-gamma)*disc_mass)/(2*np.pi*(radius_critical**2)))
# ##############
# # Sink setup #
# ##############
# stellar_mass = 1.0
# stellar_accretion_radius = 1
# stellar_position = (0.0, 0.0, 0.0)
# stellar_velocity = (0.0, 0.0, 0.0)
#
#
#
#
#
# ###########################
# # Equation of state setup #
# ###########################
# ieos = 3
# q_index = 0.25
# qfacdisc = 0.5
# T0 = 300
# Tinf = 10
# R0 = 0.25
# reference_radius = 1.0
#
#
# ################################################################################
# # Calculate the simualtion run time based on number of orbits and size of disc #
# ################################################################################
# period = (radius_max **(3/2)) # Outer orbital period in years.
# tmax = ((period * yr * norbits)/utime) # Simulation runtime in code units to follow disc for norbits
# time_between_dumps = tmax/100 # Time in code units between dumps, by default set to produce 100 dump files
#
# igas = phantomsetup.defaults.PARTICLE_TYPE['igas']
# prefix = 'disc'
# particle_type = igas
#
#
# length_unit = phantomsetup.units.unit_string_to_cgs('au')
# mass_unit = phantomsetup.units.unit_string_to_cgs('solarm')
# gravitational_constant = 1.0
#
#
# # def density_distribution(radius, radius_critical, gamma):
# #     """Self-similar disc surface density distribution.
# #
# #     This is the Lyden-Bell and Pringle (1974) solution, i.e. a power law
# #     with an exponential taper.
# #     """
# #     return phantomsetup.disc.self_similar_accretion_disc(radius, radius_critical, gamma) * sigma_crit
# #
#
# def density_distribution(radius, p_index, disc_mass,R0,radius_min,radius_max):
#     """Disc surface density distribution.
#     """
#     return phantomsetup.disc.my_surface_density(radius, p_index, disc_mass,R0,radius_min,radius_max)
#
#
# setup = phantomsetup.Setup()
# setup.prefix = prefix
# setup.set_units(
#     length=length_unit, mass=mass_unit, gravitational_constant_is_unity=True
# )
# tree_accuracy=0.3
# setup.set_compile_option('GRAVITY', True)
# setup.set_run_option('tree_accuracy', 0.3)
# setup.set_run_option('tree_accuracy', tree_accuracy)
#
#
# polyk = phantomsetup.eos.polyk_for_locally_isothermal_disc_mine(
#     T0,q_index, reference_radius, stellar_mass, gravitational_constant
# )
#
# aspect_ratio = phantomsetup.eos.get_aspect_ratio(
# T0, q_index,reference_radius,stellar_mass,gravitational_constant)
#
#
#
# setup.set_equation_of_state(ieos=ieos, polyk=polyk)
#
#
#
# setup.set_dissipation(disc_viscosity=True, alpha=alpha_artificial)
#
#
#
# setup.add_sink(
#     mass=stellar_mass,
#     accretion_radius=stellar_accretion_radius,
#     position=stellar_position,
#     velocity=stellar_velocity,
#
# )
#
#
#
# disc = phantomsetup.Disc(
#     particle_type=particle_type,
#     T0 = T0,
#     Tinf = Tinf,
#     R0 = R0,
#     number_of_particles=number_of_particles,
#     disc_mass=disc_mass,
#     density_distribution=density_distribution,
#     radius_range=(radius_min, radius_max),
#     q_index=q_index,
#     qfacdisc = qfacdisc,
#     p_index=p_index,
#     aspect_ratio=aspect_ratio,
#     reference_radius=reference_radius,
#     stellar_mass=stellar_mass,
#     gravitational_constant=gravitational_constant,
#     # extra_args=(radius_critical,gamma),
#     extra_args=(p_index,disc_mass,R0,radius_min,radius_max),
#
# )
#
# setup.add_container(disc)
#
#
#
#
#
# working_dir = cwd
#
#
#
# setup.write_dump_file(directory=working_dir)
# setup.write_in_file(directory=working_dir)
#
#
#
# print("#" * 50)
# print("               Accretion disc setup               ")
# print("-" * 50)
# print('Stellar mass             = ', stellar_mass, 'solar masses')
# print('Stellar accretion radius = ', stellar_accretion_radius, 'AU')
# print('Disc mass                = ', disc_mass, 'solar masses')
# print('Disc radius              = ', radius_max, 'AU')
# print('Number of particles      = ', number_of_particles)
# print('Number of orbits         = ', norbits)
# print("-" * 50)
# print(" Edit the disc.in file to change runtime options, ")
# print("    any values not defined are set to default     ")
# print("-" * 50)
# print("           Set the following in disc.in           ")
# print("-" * 50)
# print('tmax                     = ', tmax)
# print('dtmax                    = ', time_between_dumps)
# print("#" * 50)
