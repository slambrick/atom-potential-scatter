#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 15:16:00 2020

@author: sam
"""

from os import walk
import atom_potential_scatter as atom
import matplotlib.pyplot as plt
from numpy import max


def energy_check(f, surf, potential):
    t = atom.Trajectory.load_trajectory(f)
    dE = max(t.energy_deviation(surf, potential))

def main():
    data_path = 'results/0001-test_bump_small_potential/'
    # Loop through each of the trajectories and calculate the maximum deviation
    # of total energy 
    (_, _, fnames) = next(walk(data_path))
    fnames = [f for f in fnames if f[0:8] == 'particle']
    fnames = [data_path + f for f in fnames]
    s = atom.Surf()
    p = atom.Potential.load_potential(data_path)
    dE = [energy_check(f, s, p) for f in fnames]

    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1.2, 0.7])
    ax.histogram(dE)
    ax.set_xlabel('% Energy deviation')
    ax.set_ylabel('# of particles')
    fig.savefig(data_path + 'energy_deviation.eps')


if __name__ == '__main__':
    main()
