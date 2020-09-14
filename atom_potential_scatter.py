#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 17:24:29 2020

@author: sam

Module for interfacing with the atom-potential-scatter shared library written
in rust. Also includes a few useful pure python functions.
"""

from ctypes import c_double, c_char, c_uint64
import ctypes
import numpy as np

atom_scatter = ctypes.CDLL("atom-potential-scatter/target/release/libatom_potential_scatter.so")


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


def run_single_particle(init_pos, init_v, dt, it, fname, potential):
    """Runs a single particle with the specified initial conditions through the
    potential for a given number of iteration with the given timestep."""

    p = (c_double*3)(*[potential['Depth'], potential['Distance'],
                       potential['Width']])
    x = c_double(init_pos[0])
    y = c_double(init_pos[1])
    vx = c_double(init_v[0])
    vy = c_double(init_v[1])
    arr = (c_char*len(fname))(*fname.encode('ascii'))
    atom_scatter.single_particle(x, y, vx, vy, c_double(dt), c_uint64(it), arr,
                                 c_uint64(len(fname)), p)


def run_many_particle(init_pos, init_v, dt, it, fname, potential, record):
    p = (c_double*3)(*[potential['Depth'], potential['Distance'],
                       potential['Width']])
    xs = init_pos[0, ].ctypes.data_as(ctypes.POINTER(c_double))
    ys = init_pos[1, ].ctypes.data_as(ctypes.POINTER(c_double))
    vxs = init_v[0, ].ctypes.data_as(ctypes.POINTER(c_double))
    vys = init_v[1, ].ctypes.data_as(ctypes.POINTER(c_double))
    arr = (c_char*len(fname))(*fname.encode('ascii'))
    n_atom = c_uint64(np.shape(init_pos)[1])
    atom_scatter.multiple_particle(xs, ys, vxs, vys, n_atom, c_double(dt),
                                   c_uint64(it), arr, c_uint64(len(fname)), p,
                                   c_uint64(record))


def morse_potential(xs, ys, potential):
    """Uses the implementation of the morse potential functioin in rust to
    evaluate it for the given numpy arrays of x,y."""

    p = (c_double*3)(*[potential['Depth'], potential['Distance'],
                       potential['Width']])
    n = xs.shape[0]
    i = xs.ctypes.data_as(ctypes.POINTER(c_double))
    j = ys.ctypes.data_as(ctypes.POINTER(c_double))
    V = np.zeros(n)
    k = V.ctypes.data_as(ctypes.POINTER(c_double))
    atom_scatter.calc_potential(i, j, k, c_uint64(n), p)
    return(V)


def morse_1d(ys, potential):
    """Calculates the morse potential for the given parameters and y
    position(s)"""

    de = potential['Depth']
    a = potential['Width']
    re = potential['Distance']
    return(de*(np.exp(-2.0*a*(ys - re)) - 2.0*np.exp(-a*(ys - re))))
