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
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow
from PyQt5.QtCore import Qt

from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

plt.style.use('ggplot')


# Probably keep this one?
def plot_potential_1d(potential):
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
    conservation of energy for that trajectory."""

    # Generate a random surface
    h_RMS = 0.5
    corr = 10
    Dx = 0.05
    surf = atom.Surf()#atom.RandSurf.random_surf_gen(h_RMS, Dx, corr, 5001)

    # Set the parameters of the potential
    De = 0.2
    re = 1.0
    a = 1.0
    potential = atom.Potential(De, re, a)

    init_pos = [-15, 12]
    init_v = [1, -1]/np.sqrt(2)
    dt = 0.001
    it = 5800*5
    d = atom.run_single_particle(init_pos, init_v, dt, it, fname, potential,
                                 surf=surf)
    traj = atom.Trajectory.load_trajectory(d)
    fig1, ax1 = atom.plot_potential_2d(surf, potential, figsize=[0, 0, 1.0, 0.6])
    _, _ = traj.plot_traj(surf, potential, fig=fig1, ax=ax1)
    ax1.set_ylim([-2, 15])
    ax1.set_xlim([-25, 25])
    fig2, ax2 = traj.error_plot()
    fig3, ax3 = traj.conservation_of_energy(surf, potential)
    dv = traj.energy_deviation(surf, potential)
    fig4 = plt.figure()
    ax4 = fig4.add_axes([0, 0, 1.0, 0.6])
    ax4.plot(traj.times, dv)
    ax4.set_xlabel('Time')
    ax4.set_ylabel('% energy deviation')
    print(traj.cuml_error())

# TODO: update
def many_single_particles_test(save_dir):
    """Runs through many particles in the test potential and consideres the
    conservation of energy of thise particles as a test case. By default records
    all iterations of all trajectories and is non-parallel. This is designed
    for debuging and testing on medium scale simulations. Use run_many_particle
    for doing full scale simulations to avoid excessive sized data files and
    computation time."""

    h_RMS = 1
    corr = 10
    Dx = 0.02
    surf = atom.RandSurf()
    surf.random_surf_gen(h_RMS, Dx, corr, 5001)

    # Set the parameters of the potential
    De = 0.5
    a = 1
    re = atom.ye_from_De(De, a)
    potential = atom.Potential(De, re, a)

    init_xs = np.linspace(-40, 10, 51)
    init_y = 15
    init_v = [1, -1]
    dt = 0.002/5
    it = 3500*5*5

    save_dir = atom._simulation_dir(save_dir)
    potential.save_potential(save_dir)
    # Run the simulations
    start = time.time()
    fnames = []
    for i, x in enumerate(init_xs):
        fname = save_dir + '/' + 'particle' + str(i).zfill(1) + '.csv'
        if os.path.isfile(fname):
            os.remove(fname)
        atom.run_single_particle([x, init_y], init_v, dt, it, fname, potential,
                                 surf=surf, new_dir=False)
        fnames.append(fname)
    end = time.time()
    print(end- start)

    (fig, ax) = atom.plot_potential_2d(surf, potential, figsize=[0, 0, 1.0, 0.6])
    trajs = []
    for f in fnames:
        t = atom.Trajectory.load_trajectory(f)
        trajs.append(t)
        _, _ = t.plot_traj(surf, potential, fig=fig, ax=ax)
    ax.set_xlim([-40, 40])
    ax.set_ylim([-2, 20])
    ax.set_title('')
    fig.savefig(save_dir + '/trajectories_gayss_bump.eps', bbox_inches="tight")
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
    speed = 1
    cond = atom.Conditions(n_atom, dt, it)
    cond.set_position([-50, 50], 15)
    cond.set_velocity(init_angle, speed)

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


# Subclass QMainWindow to customise your application's main window
class MainWindow(QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        self.setWindowTitle("My Awesome App")

        label = QLabel("This is a PyQt5 window!")

        # The `Qt` namespace has a lot of attributes to customise
        # widgets. See: http://doc.qt.io/qt-5/qt.html
        label.setAlignment(Qt.AlignCenter)

        # Set the central widget of the Window. Widget will expand
        # to take up all the space in the window by default.
        self.setCentralWidget(label)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()  # IMPORTANT!!!!! Windows are hidden by default.
    app.exec_()


if __name__ == '__main__':
    many_single_particles_test('test_random')
    #ts = single_particle_test('verlet_method')
