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


def main():
    multiple_particle_test()


if __name__ == "__main__":
    main()
