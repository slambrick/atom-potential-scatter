#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 12:36:01 2020

@author: sam
"""

# matplotlib import has loads of warnings
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import os
import atom_potential_scatter as atom
import time

from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

plt.style.use('ggplot')


def plot_potential_2d(potential):
    xx = np.linspace(-50, 50, 401)
    yy = np.linspace(-2, 30, 171)
    g = np.meshgrid(xx, yy)
    V = atom.morse_potential(g[0].flatten(), g[1].flatten(), potential).reshape((171, 401))
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 2, 0.5])
    cs = ax.contourf(g[0], g[1], V, cmap=cm.coolwarm,
                     levels=np.linspace(-1, 1, 21), extend='max')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.axis('equal')
    fig.colorbar(cs, ax=ax)
    ax.set_title("Atom potential")
    return(fig, ax)


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
    potential = {'Depth': 0.5, 'Distance': 0.5, 'Width': 1}
    if os.path.isfile(fname):
        os.remove(fname)
    dt = 0.01
    it = 2800
    atom.run_single_particle([-25, 15], [1, -1], dt, it, fname, potential)
    df = pd.read_csv("can_I_tell_you_where_to_save.csv")
    (fig, ax) = plot_potential_2d()
    ax.plot(df['x'], df['y'], color='black')
    ax.set_ylim([-1, 15])
    ax.set_xlim([-40, 40])
    ax.set_title("Atom potential and trajectory")
    fig.savefig("one_trajectory.pdf", bbox_inches="tight")


def many_single_particles_test(save_dir):
    potential = {'Depth': 0.5, 'Distance': 0.5, 'Width': 1}
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    init_v = [1, -1]
    dt = 0.01
    it = 3300
    init_xs = np.linspace(-40, 10, 51)
    y = 15

    # Run the simulations
    fnames = []
    for i, x in enumerate(init_xs):
        fname = save_dir + '/' + 'particle' + str(i).zfill(1) + '.csv'
        if os.path.isfile(fname):
            os.remove(fname)
        atom.run_single_particle([x, y], init_v, dt, it, fname, potential)
        fnames.append(fname)

    (fig, ax) = plot_potential_2d(potential)

    # Plot the trajectories
    for f in fnames:
        df = pd.read_csv(f)
        ax.plot(df['x'], df['y'], color='black')
    ax.set_ylim([-1, 20])
    ax.set_xlim([-40, 40])
    ax.set_title("Atom potential and trajectory")
    fig.savefig("many_trajectories.pdf", bbox_inches="tight")


def trajectory_plot(dir_name, potential):
    d = "results/" + dir_name if dir_name[0:7] != "results" else dir_name
    # Plot the trajectories
    (fig, ax) = plot_potential_2d(potential)
    fnames = os.listdir(path=d)
    fnames2 = list(filter(lambda k: "atom" in k, fnames))
    for f in fnames2:
        df = pd.read_csv(dir_name + "/" + f)
        ax.plot(df['x'], df['y'], color='black')
    ax.set_ylim([-1, 20])
    ax.set_xlim([-40, 40])
    ax.set_title("Atom potential and trajectory")
    fig.savefig(d + "/many_trajectories_rust.pdf", bbox_inches="tight")


def error_plot(dir_name):
    d = "results/" + dir_name if dir_name[0:7] != "results" else dir_name
    fnames = os.listdir(path=d)
    fnames2 = list(filter(lambda k: "atom" in k, fnames))
    # Plot the errors
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1.2, 1])
    for f in fnames2:
        df = pd.read_csv(d + "/" + f)
        ax.plot(df['time'], df['e_x'], color='blue')
        ax.plot(df['time'], df['e_y'], color='red')
    ax.legend(['Error in x'], ['Error in y'])

    fig2 = plt.figure()
    ax = fig2.add_axes([0, 0, 1.2, 1])
    cuml_errors = np.zeros(len(fnames2))
    for i, f in enumerate(fnames2):
        df = pd.read_csv(d + "/" + f)
        cuml_errors[i] = sum(df['e_x'])
    ax.hist(cuml_errors)


def multiple_particle_test():
    n_atom = 10001
    x = np.linspace(-40, 10, n_atom)
    y = np.repeat(15, n_atom)
    init_pos = np.array([x, y])
    init_v = np.repeat(np.array([[1.0], [-1.0]]), n_atom, axis=1)
    dt = 0.02
    it = 1650
    fname = "test"
    potential = {'Depth': 0.8, 'Distance': 0.5, 'Width': 1}
    record = 150
    start = time.time()
    d = atom.run_many_particle(init_pos, init_v, dt, it, fname, potential,
                               record, method="Fehlberg")
    end = time.time()
    print(end - start)
    trajectory_plot(d, potential)
    error_plot(d)


def plot_random_surface(surf, potential):
    fig1 = plt.figure()
    ax1 = fig1.add_axes([0, 0, 1, 0.5])
    ax1.plot(surf['x'], surf['y'])

    xx = np.linspace(-80, 80, 401)
    yy = np.linspace(-5, 30, 171)
    g = np.meshgrid(xx, yy)
    V = atom.morse_potential(g[0].flatten(), g[1].flatten(), potential, surf).reshape((171, 401))
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 3, 0.5])
    cs = ax.contourf(g[0], g[1], V, cmap=cm.coolwarm,
                     levels=np.linspace(-1, 1, 21), extend='max')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(cs, ax=ax)
    ax.set_title("Atom potential")
    return(fig, ax)


def test_surf_gen():
    # Generate a random surface
    h_RMS = 1
    corr = 10
    Dx = 0.02
    surface = atom.Surf(h_RMS, Dx, corr, 100001)
    lst_ax = surface.plot_surf_properties()
    # TODO: make these all as one figure in python
    lst_ax[0][0].savefig("surface_profile.pdf", bbox_inches="tight")
    lst_ax[1][0].savefig("height_distribution.pdf", bbox_inches="tight")
    lst_ax[2][0].savefig("correlation_function.pdf", bbox_inches="tight")
    lst_ax[3][0].savefig("gradient_distribution.pdf", bbox_inches="tight")


def potential_and_trajectory_plot(d, surf, potential, n_atom, record):
    # Plot the potential
    (fig, ax) = plot_random_surface(surf.as_dict(), potential)
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


def final_direction_plot(d, y_cutoff):
    df = pd.read_csv(d + '/final_states.csv')
    df_ended = df[np.logical_and(df['y'] > y_cutoff,
                                 np.logical_not(np.isnan(df['y'])))]
    n_missed = len(df['x']) - len(df_ended['x'])
    df_ended['theta'] = np.arctan(df_ended['v_x']/df_ended['v_y'])*180/np.pi
    print(n_missed)
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 0.7, 0.7])
    ax.hist(df_ended['theta'], density=True)
    ax.set_xlim([-90, 90])
    ax.set_xlabel(r'$\theta/^o$')
    ax.set_ylabel('Probability')
    fig.savefig(d + '/final_scattering_distribution.pdf', bbox_inches='tight')
    return(df_ended)


def test_random_scatter():
    h_RMS = 0.05
    corr = 10
    Dx = 0.04
    # Generate a random surface
    surf = atom.Surf(h_RMS, Dx, corr, 10001)

    # Trace atom trajectories
    n_atom = 2001
    x = np.linspace(-50, 50, n_atom)
    y = np.repeat(15, n_atom)
    init_angle = 20
    speed = 1
    init_pos = np.array([x, y])
    v = speed*np.array([[np.sin(init_angle*np.pi/180)],
                        [-np.cos(init_angle*np.pi/180)]])
    init_v = np.repeat(v, n_atom, axis=1)
    dt = 0.05
    it = 900
    fname = "roughness_investigation1_"
    potential = {'Depth': 0.2, 'Distance': 0.5, 'Width': 1}
    init_conditions = {'n atom': n_atom, 'Position': init_pos,
                       'Velocity': init_v, 'Time step': dt, 'Iterations': it}
    record = 1
    start = time.time()
    d = atom.run_many_particle(init_pos, init_v, dt, it, fname, potential,
                               record, surf=surf.as_dict(), method="Fehlberg")
    end = time.time()
    print(end - start)
    surf.save_surf(d)
    atom.save_potential(potential, d)
    atom.save_initial_conditions(init_conditions, d)
    potential_and_trajectory_plot(d, surf, potential, n_atom, record)
    final_direction_plot(d, 10.0)
    print('Data is stored in: ', d)
    return(d, surf, potential)


def main():
    test_random_scatter()


if __name__ == "__main__":
    main()
