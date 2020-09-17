#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 17:24:29 2020

@author: sam

Module for interfacing with the atom-potential-scatter shared library written
in rust. Also includes a few useful pure python functions. Interfacing is done
via the c_types module.
"""

from ctypes import c_double, c_char, c_uint64, POINTER
import ctypes
import numpy as np
import os

atom_scatter = ctypes.CDLL(("atom-potential-scatter/target/release/libatom_pot"
                            "ential_scatter.so"))

DEFAULT_SURF = {'x': np.array([]), 'y': np.array([]), 'type': 'gauss'}


def random_surf_gen_crude(h, Dx, lambd, s, N=1000):
    Z = np.random.normal(0.0, s, N)
    f = np.zeros(N)
    for n in range(N):
        ms = np.linspace(0, N-1, N)
        f[n] = sum(Z * np.exp(-abs(n - ms)*Dx/lambd))
    xs = np.linspace(0, Dx*(N-1), N)
    return(h*f, xs)


def random_surf_gen_core(h, Dx, lambd, s, N=10001):
    """Core routine to generate a random surface. Arguments should be chosen
    carefully. The number of points must be odd."""

    if N % 2 != 1:
        raise ValueError("Must be an odd number of points in the surface")
    Z = np.random.normal(0.0, s, N)
    ms = np.linspace(-round(N/2), round(N/2), N)
    e = np.exp(-abs(ms)*Dx/lambd)
    f = np.convolve(Z, e, 'same')
    xs = ms*Dx
    return(f, xs)


def simulation_dir(name="just_a_test"):
    """Creates a name for a directory for the simultion being performed.
    Automatically numbers the directory with a 4 digit number and then creates
    that directory in 'results' and returns the full path."""

    if name[0] == "/":
        name = name[1:]
    if not os.path.isdir("results"):
        os.mkdir("results")
    past = list(map(lambda x: "results/" + x, os.listdir("results")))
    past_dirs = list(filter(os.path.isdir, past))
    inds = list(map(int, list(map(lambda s: s[-4:], past_dirs))))
    ind = 1 if len(past_dirs) == 0 else max(inds) + 1
    dir_name = "results/" + name + str(ind).zfill(4)
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)
    return(dir_name)


def passing_string_to_rust():
    """Demo function for passing strings from python to rust. Not really
    usefull for anything"""

    L = "Hello from rust!"
    arr = (c_char*len(L))(*L.encode('ascii'))
    atom_scatter.print_text(arr, c_uint64(len(L)))


def gauss_bump(xs):
    """Uses the rust function to evaluate the gaussian bump test model of the
    surface."""

    n = xs.shape[0]
    i = xs.ctypes.data_as(ctypes.POINTER(c_double))
    ys = np.zeros(n)
    j = ys.ctypes.data_as(ctypes.POINTER(c_double))
    atom_scatter.gauss_bump(i, j, c_uint64(n), c_double(10))
    return(ys)


def run_single_particle(init_pos, init_v, dt, it, sim_name, potential,
                        surf=DEFAULT_SURF, method="Fehlberg"):
    """Runs a single particle with the specified initial conditions through the
    potential for a given number of iteration with the given timestep."""

    # Put the surface information into the correct types
    t = surf['type'] == 'interpolate'
    surf_x = surf['x'].ctypes.data_as(POINTER(c_double)) if t else (c_double*0)(*[])
    surf_y = surf['y'].ctypes.data_as(POINTER(c_double)) if t else (c_double*0)(*[])
    test_surf = c_uint64(0) if t else c_uint64(1)
    surf_n = c_uint64(len(surf['x'])) if t else c_uint64(0)

    # Put the potential values into a C array of doubles
    p = (c_double*3)(*[potential['Depth'], potential['Distance'],
                       potential['Width']])
    # Put the initial conditions into C arrays of doubles
    x = c_double(init_pos[0])
    y = c_double(init_pos[1])
    vx = c_double(init_v[0])
    vy = c_double(init_v[1])

    # Put the directory name and integration method into a C array of chars
    d = simulation_dir(sim_name)
    arr = (c_char*len(d))(*d.encode('ascii'))
    meth = (c_char*len(method))(*method.encode('ascii'))
    atom_scatter.single_particle(x, y, vx, vy, c_double(dt), c_uint64(it), arr,
                                 c_uint64(len(d)), p, meth, c_uint64(len(meth)), 
                                 surf_x, surf_y, surf_n, test_surf)


def run_many_particle(init_pos, init_v, dt, it, sim_name, potential, record,
                      surf=DEFAULT_SURF, method="Fehlberg"):
    # Put the surface information into the correct types
    t = surf['type'] == 'interpolate'
    surf_x = surf['x'].ctypes.data_as(POINTER(c_double)) if t else (c_double*0)(*[])
    surf_y = surf['y'].ctypes.data_as(POINTER(c_double)) if t else (c_double*0)(*[])
    test_surf = c_uint64(0) if t else c_uint64(1)
    surf_n = c_uint64(len(surf['x'])) if t else c_uint64(0)

    d = simulation_dir(sim_name)
    # Put the potential values into a C array of doubles
    p = (c_double*3)(*[potential['Depth'], potential['Distance'],
                       potential['Width']])
    # Put the initial conditions into C arrays of doubles
    xs = init_pos[0, ].ctypes.data_as(POINTER(c_double))
    ys = init_pos[1, ].ctypes.data_as(POINTER(c_double))
    vxs = init_v[0, ].ctypes.data_as(POINTER(c_double))
    vys = init_v[1, ].ctypes.data_as(POINTER(c_double))

    # Put the directory name and integration method into a C array of chars
    d = simulation_dir(sim_name)
    arr = (c_char*len(d))(*d.encode('ascii'))
    meth = (c_char*len(method))(*method.encode('ascii'))

    # Number of atoms we are simulating as a C int
    n_atom = c_uint64(np.shape(init_pos)[1])
    atom_scatter.multiple_particle(xs, ys, vxs, vys, n_atom, c_double(dt),
                                   c_uint64(it), arr, c_uint64(len(d)), p,
                                   c_uint64(record), meth, c_uint64(len(meth)),
                                   surf_x, surf_y, surf_n, test_surf)
    return(d)


def morse_potential(xs, ys, potential, surf=DEFAULT_SURF):
    """Uses the implementation of the morse potential functioin in rust to
    evaluate it for the given numpy arrays of x,y."""

    # Put the surface information into the correct types
    t = surf['type'] == 'interpolate'
    surf_x = surf['x'].ctypes.data_as(POINTER(c_double)) if t else (c_double*0)(*[])
    surf_y = surf['y'].ctypes.data_as(POINTER(c_double)) if t else (c_double*0)(*[])
    test_surf = c_uint64(0) if t else c_uint64(1)
    surf_n = c_uint64(len(surf['x'])) if t else c_uint64(0)
    print(test_surf)

    p = (c_double*3)(*[potential['Depth'], potential['Distance'],
                       potential['Width']])
    n = xs.shape[0]
    i = xs.ctypes.data_as(POINTER(c_double))
    j = ys.ctypes.data_as(POINTER(c_double))
    V = np.zeros(n)
    k = V.ctypes.data_as(POINTER(c_double))
    atom_scatter.calc_potential(i, j, k, c_uint64(n), p, surf_x, surf_y,
                                surf_n, test_surf)
    return(V)


def morse_1d(ys, potential):
    """Calculates the morse potential for the given parameters and y
    position(s)"""

    de = potential['Depth']
    a = potential['Width']
    re = potential['Distance']
    return(de*(np.exp(-2.0*a*(ys - re)) - 2.0*np.exp(-a*(ys - re))))


def interp_test(x, y, xs):
    i = x.ctypes.data_as(ctypes.POINTER(c_double))
    j = y.ctypes.data_as(ctypes.POINTER(c_double))
    k = xs.ctypes.data_as(ctypes.POINTER(c_double))
    ys = np.zeros(xs.shape[0])
    ll = ys.ctypes.data_as(ctypes.POINTER(c_double))
    atom_scatter.interpolate_test(i, j, c_uint64(x.shape[0]), k, ll,
                                  c_uint64(xs.shape[0]))
    return(ys)
