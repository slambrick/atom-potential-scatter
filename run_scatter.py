#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 12:36:01 2020

@author: sam
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import os
import atom_potential_scatter as atom
import time
import sys
import tkinter as tk

from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

plt.style.use('ggplot')


# Probably keep this one? Move to the module?
def plot_potential_1d(potential):
    """Plots a potential provided in the potential object."""
    
    y = np.linspace(-1, 5, 501)
    V = atom.morse_1d(y, potential)
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1.2, 1])
    ax.plot(y, V)
    ax.set_xlabel('Height from surface')
    ax.set_ylabel('Potential')
    ax.set_ylim(-1, 2)
    ax.set_xlim(-1, 4)
    return(fig, ax)


def single_particle_test(fname):
    """Runs a single particle through the test potential and checks the
    conservation of energy for that trajectory. Note that this function uses
    the debug surface consisting of a single gaussian 'defect'."""

    # Generate a random surface
    h_RMS = 0.5 # The RMS height for the generated surface (nm)
    corr = 10   # The correlation length for the generated surface (nm)
    Dx = 0.05   # The element length in the surface (nm)
    surf = atom.Surf() # Default surface: Gaussian bump

    # Set the parameters of the potential
    De = 1.0    # Depth
    re = 1.0    # Displacement (nm)
    a = 1.0     # Width (nm)
    potential = atom.Potential(De, re, a)
    
    # Iterations
    dt = 0.001
    it = 5800*3
    
    # Initial condition object
    init_cond = atom.Conditions(n_atom=1, Dt=dt, n_it=it, height_stop=10)
    init_cond.set_velocity(45)
    init_cond.set_position([-10, 10], 12)
    
    # Run the simulation
    d, save_dir = atom.run_single_particle(init_cond, fname, potential, surf=surf)
    
    # Load and plot the trajectory
    traj = atom.Trajectory.load_trajectory(d)
    fig1, ax1 = atom.plot_potential_2d(surf, potential, figsize=(7, 3))
    _, _ = traj.plot_traj(surf, potential, fig=fig1, ax=ax1)
    ax1.set_ylim([-2, 15])
    ax1.set_xlim([-25, 25])
    
    # Produce a plot of errors in the positions and velocities (compared to an
    # analytic model)
    fig2, ax2 = traj.error_plot()
    
    # Produce plots of energy conservation
    fig3, ax3 = traj.conservation_of_energy(surf, potential)
    dv = traj.energy_deviation(surf, potential)
    fig4 = plt.figure(figsize=(6,4))
    ax4 = fig4.add_axes([0.15, 0.15, 0.8, 0.8])
    ax4.plot(traj.times, dv)
    ax4.set_xlabel('Time')
    ax4.set_ylabel('% energy deviation')
    print(traj.cuml_error())
    
    # Save all the plots as .eps and .png
    plt.tight_layout()
    fig1.savefig(save_dir + '/trajectories.pdf', bbox_inches="tight")
    fig1.savefig(save_dir + '/trajectories.png', dpi=300, bbox_inches="tight")
    fig2.savefig(save_dir + '/trajectory_error_plot.pdf', bbox_inches="tight")
    fig2.savefig(save_dir + '/trajectory_error_plot.png', dpi=300, bbox_inches="tight")
    fig3.savefig(save_dir + '/trajectory_energy.pdf', bbox_inches="tight")
    fig3.savefig(save_dir + '/trajectory_energy.png', dpi=300, bbox_inches="tight")
    fig4.savefig(save_dir + '/energy_deviation.pdf', bbox_inches="tight")
    fig4.savefig(save_dir + '/energy_deviation.png', dpi=300, bbox_inches="tight")
    

def many_single_particles_test(save_dir, bump=False):
    """Runs through many particles in the test potential and consideres the
    conservation of energy of these particles as a test case. By default records
    all iterations of all trajectories and is non-parallel. This is designed
    for debuging and testing on medium scale simulations. Use run_many_particle
    for doing full scale simulations to avoid excessive sized data files and
    computation time."""

    h_RMS = 0.5
    corr = 10
    Dx = 0.02
    surf = atom.RandSurf()
    if not bump:
        surf.random_surf_gen(h_RMS, Dx, corr, 5001)

    # Set the parameters of the potential
    De = 0.5    # Depth
    a = 1     # Width (nm)
    re = atom.ye_from_De(De, a)     # Displacement (nm)
    potential = atom.Potential(De, re, a)

    # Initial conditions for the atoms
    x_range = (-40, 10) # Range of starting x positions
    n_atom = 21         # Number of atoms
    init_y = 12         # Initial y coordinate
    dt = 0.002          # Timestep
    it = 3500*20         # Max number of iterations
    init_cond = atom.Conditions(n_atom=n_atom, Dt=dt, n_it=it,
                                height_stop=10)
    init_cond.set_position(x_range, init_y)
    init_cond.set_velocity(45)

    save_dir = atom._simulation_dir(save_dir)
    potential.save_potential(save_dir)
    surf.save_surf(save_dir)
    init_cond.save_inital_conditions(save_dir)
    
    # Run the simulations
    start = time.time()
    fnames = []
    for i, x in enumerate(init_cond.position[0,]):
        fname = save_dir + '/' + 'particle' + str(i).zfill(1) + '.csv'
        if os.path.isfile(fname):
            os.remove(fname)
        init_cond_tmp = init_cond.get_atom_n(i)
        atom.run_single_particle(init_cond_tmp, fname, potential, surf=surf, new_dir=False)
        fnames.append(fname)
    end = time.time()
    print(end- start)

    (fig, ax) = atom.plot_potential_2d(surf, potential, figsize=(8,4))
    trajs = []
    for f in fnames:
        t = atom.Trajectory.load_trajectory(f)
        trajs.append(t)
        _, _ = t.plot_traj(surf, potential, fig=fig, ax=ax)
    ax.set_xlim([-40, 40])
    ax.set_ylim([-2, 20])
    ax.set_title('')
    fig.savefig(save_dir + '/trajectories.eps', bbox_inches="tight")
    return(trajs)


def many_partlce_test(save_dir):
    # Generate a random surface
    h_RMS = 1
    corr = 10
    Dx = 0.02
    surface = atom.RandSurf()
    surface.random_surf_gen(h_RMS, Dx, corr, 5001)

    # Set the parameters of the potential
    De = 0.001
    a = 1.5
    re = atom.ye_from_De(De, a)
    potential = atom.Potential(De, re, a)

    # Set the initial conditions
    n_atom = 101
    it = 3500*5*5
    dt = 0.002/5
    init_angle = 40
    cond = atom.Conditions(n_atom, dt, it)
    cond.set_position([-50, 50], 15)
    cond.set_velocity(init_angle)

    # What proportion of trajectories should be saved (1 in every n)
    n_record = 1

    # Run the simulations
    start = time.time()
    d = atom.run_many_particle(cond, save_dir, potential, n_record, surface,
                               method="Verlet")
    end = time.time()
    print(end - start)

    # Save the parameters to the same file
    potential.save_potential(d)
    cond.save_inital_conditions(d)

    fnames = []
    for i in range(n_atom):
        fname = d + '/' + 'atom' + str(i).zfill(8) + '.csv'
        if os.path.isfile(fname):
            os.remove(fname)
        fnames.append(fname)
    end = time.time()
    print(end- start)

    # Produce plots of the potential and the trajectories
    (fig, ax) = atom.plot_potential_2d(surface, potential, figsize=[0, 0, 1.0, 0.6])
    trajs = []
    for f in fnames[::10]:
        t = atom.Trajectory.load_trajectory(f)
        trajs.append(t)
        _, _ = t.plot_traj(surf, potential, fig=fig, ax=ax)
    ax.set_xlim([-40, 40])
    ax.set_ylim([-2, 20])
    ax.set_title('')
    final_direction_plot(d, 10.0, potential, surface)
    print('Data is stored in: ', d)
    return(d, surf, cond, potential)

def test_surf_gen():
    """Tests surface generation and saves the resulting statistics plots."""

    # Generate a random surface
    h_RMS = 1
    corr = 10
    Dx = 0.02
    surface = atom.RandSurf()
    surface.random_surf_gen(h_RMS, Dx, corr, 100001)
    lst_ax = surface.plot_surf_properties()
    # TODO: make these all as one figure in python
    lst_ax[0][0].savefig("surface_profile.pdf", bbox_inches="tight")
    lst_ax[1][0].savefig("height_distribution.pdf", bbox_inches="tight")
    lst_ax[2][0].savefig("correlation_function.pdf", bbox_inches="tight")
    lst_ax[3][0].savefig("gradient_distribution.pdf", bbox_inches="tight")


# TODO: move into the other module
def potential_and_trajectory_plot(d, surf, potential, n_atom, record):
    # Plot the potential
    (fig, ax) = atom.plot_potential_2d(surf, potential)
    fnames = os.listdir(path=d)
    fnames2 = list(filter(lambda k: "atom" in k, fnames))
    # Add some of the trajectories
    n_plot = round(n_atom/record)
    skip = 1 if n_plot < 100 else round(n_plot/100)
    for i, f in enumerate(fnames2):
        if i % skip == 0:
            df = pd.read_csv(d + "/" + f)
            ax.plot(df['x'], df['y'], color='black')
    ax.set_ylim([-5, 20])
    ax.set_xlim([-80, 80])
    ax.set_title("Atom potential and trajectory")
    fig.savefig(d + "/many_trajectories_rust.pdf", bbox_inches="tight")


def test_random_scatter():
    """Interactive function for running a simulation of a rough surface."""

    # Generate a random surface
    h_RMS = 0.5
    corr = 10
    Dx = 0.04
    surf = atom.RandSurf()
    surf.random_surf_gen(h_RMS, Dx, corr, 1801)

    # Set the initial conditions
    n_atom = 101
    n_it = 30000
    dt = 0.001
    init_angle = 40
    speed = 1
    cond = atom.Conditions(n_atom, dt, n_it)
    cond.set_position([-50, 50], 15)
    cond.set_velocity(init_angle, speed)

    # Set the parameters of the potential
    De = 0.5
    re = 0.0
    a = 1.0
    potential = atom.Potential(De, re, a)

    # The name of the directory to save the parameters and results in
    fname = "full_test"

    # What proportion of trajectories should be saved (1 in every n)
    n_record = 1

    # Run the simulations
    start = time.time()
    d = atom.run_many_particle(cond, fname, potential, n_record, surf, method="Verlet")
    end = time.time()
    print(end - start)

    # Save the parameters to the same file
    surf.save_surf(d)
    potential.save_potential(d)
    cond.save_inital_conditions(d)

    #final_direction_plot(d, 10.0, potential, surf)
    print('Data is stored in: ', d)
    return(d, surf, cond, potential)

# IMPORTANT
# 
# Mass of the helium atom is set to be 1
# Units of distance are in nm
# Set the kinetic energy of the atoms to be 1 for 300K He-4 atoms
# Therefore the speed of the atoms is sqrt(2) for 300K He-4 atoms
# This sets the arbitary units of time: 8.0116735e-13 units/s
# Note that this ties the ...

traj = many_single_particles_test('bump_test')
#single_particle_test('test_correct_energy')

#if __name__ == '__main__':
#    #single_particle_test('test_one_particle')
#    many_single_particles_test('test_multiple_particles')
