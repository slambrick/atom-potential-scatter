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
import matplotlib.pyplot as plt
import matplotlib.cm as cm

atom_scatter = ctypes.CDLL(("atom-potential-scatter/target/release/libatom_pot"
                            "ential_scatter.so"))

DEFAULT_SURF = {'x': np.array([]), 'y': np.array([]), 'type': 'gauss'}


class Surf:
    def __init__(self, h_RMS, Dx, corr_len, N):
        if corr_len/Dx < 10:
            raise ValueError(("The segment size Dx is not sufficently small "
                              "compared to the correlation length"))
        self.lambd = 0.5*corr_len**(2/3)
        self.h = h_RMS*np.sqrt(np.tanh(Dx/self.lambd))
        self.Dx = Dx
        (self.f, self.x) = Surf.__random_surf_gen_core(self.h, self.Dx,
                                                       self.lambd, 1, N)
        self.h_RMS = h_RMS
        self.corr_len = corr_len

    def __random_surf_gen_core(h, Dx, lambd, s, N=10001):
        """Core routine to generate a random surface. Arguments should be
        chosen carefully. The number of points must be odd."""

        if N % 2 != 1:
            raise ValueError("Must be an odd number of points in the surface")
        Z = np.random.normal(0.0, s, N)
        ms = np.linspace(-round(N/2), round(N/2), N)
        e = np.exp(-abs(ms)*Dx/lambd)
        f = h*np.convolve(Z, e, 'same')
        xs = ms*Dx
        return(f, xs)

    def plot_surf_properties(self):
        fig = plt.figure()
        ax = fig.add_axes([0, 0, 1.2, 0.35])
        ax.plot(self.x, self.f)
        ax.set_title('Generated surface profile')
        ax.set_xlabel('x/nm')
        ax.set_ylabel('y/nm')
        ax.set_xlim([-self.Dx*5000, self.Dx*5000])

        fig2 = plt.figure()
        ax2 = fig2.add_axes([0, 0, 0.6, 0.6])
        ax2.hist(self.f, density=True, label='Generated')
        end = max(abs(self.f))
        xx = np.linspace(-end, end, 501)
        yy = 1/(np.sqrt(2*np.pi)*self.h_RMS) * np.exp(-xx**2/(2*self.h_RMS**2))
        ax2.plot(xx, yy, label='Predicted')
        ax2.set_xlabel('Heights/nm')
        ax2.set_ylabel('Probability')
        ax2.set_title('Height distribution')
        ax2.legend(loc='lower right')

        fig3 = plt.figure()
        ax3 = fig3.add_axes([0, 0, 0.6, 0.6])
        auto_corr = np.correlate(self.f, self.f, mode='same')
        auto_corr = auto_corr/max(auto_corr)
        corr_x = np.arange(-len(auto_corr)*self.Dx/2, len(auto_corr)*self.Dx/2,
                           self.Dx)
        ax3.plot(corr_x, auto_corr, label='Generated')
        ax3.plot(corr_x, np.exp(-abs(corr_x)/self.lambd)*(1 + abs(corr_x)/self.lambd),
                 label='Predicted')
        ax3.set_xlabel('x/nm')
        ax3.set_ylabel('Correleation')
        ax3.set_xlim(0, 2*self.corr_len)
        ax3.set_ylim(0, 1)
        ax3.set_title('Autocorrelation function')
        ax3.legend()

        fig4 = plt.figure()
        ax4 = fig4.add_axes([0, 0, 0.6, 0.6])
        grad = np.diff(self.f)/self.Dx
        ax4.hist(grad, density=True, label='Generated')
        end = max(abs(grad))
        xx = np.linspace(-end, end, 501)
        yy = (self.lambd/(np.sqrt(2*np.pi)*self.h_RMS))*np.exp(-self.lambd**2 * xx**2/(2*self.h_RMS**2))
        ax4.plot(xx, yy, label='Predicted')
        ax4.set_xlabel('Gradients')
        ax4.set_ylabel('Probability')
        ax4.set_title('Gradient distribution')
        ax4.legend(loc='lower right')

        return([(fig, ax), (fig2, ax2), (fig3, ax3), (fig4, ax4)])

    def as_dict(self):
        return({'x': self.x, 'y': self.f, 'type': 'interpolate'})

    def plot_potential(self, param):
        fig1 = plt.figure()
        ax1 = fig1.add_axes([0, 0, 1.2, 0.5])
        ax1.plot(self.x, self.f)
        ax1.set_xlim([-self.Dx*1000, self.Dx*1000])
        ax1.set_xlabel('x/nm')
        ax1.set_ylabel('y/nm')

        xx = np.linspace(-80, 80, 401)
        yy = np.linspace(-5, 30, 171)
        g = np.meshgrid(xx, yy)
        V = morse_potential(g[0].flatten(), g[1].flatten(), param,
                            self.as_dict()).reshape((171, 401))
        fig2 = plt.figure()
        ax2 = fig2.add_axes([0, 0, 1.2, 0.5])
        cs = ax2.contourf(g[0], g[1], V, cmap=cm.coolwarm,
                          levels=np.linspace(-1, 1, 21), extend='max')
        ax2.set_xlabel("x/nm")
        ax2.set_ylabel("y/nm")
        fig2.colorbar(cs, ax=ax2)
        ax2.set_xlim([-self.Dx*1000, self.Dx*1000])
        ax2.set_ylim([-param['Distance']*4, param['Width']*10])
        ax2.set_title("Atom potential")
        return([(fig1, ax1), (fig2, ax2)])

    def save_surf(self, dir_name):
        fname = dir_name + '/surface_used.csv'
        fid = open(fname, 'w')
        fid.write('Surface statistics:\n')
        fid.write('h_RMS = ' + str(self.h_RMS) + '\n')
        fid.write('correlation_length = ' + str(self.corr_len) + '\n')
        fid.write('Surface generation parameters\n')
        fid.write('h = ' + str(self.h) + '\n')
        fid.write('Dx = ' + str(self.Dx) + '\n')
        fid.write('lambda = ' + str(self.lambd) + '\n')
        fid.write('Surface points in space:\n')
        fid.write('x,y\n')
        for x, y in zip(self.x, self.f):
            fid.write('{},{}\n'.format(x, y))
        fid.close()


def save_potential(potential, dir_name):
    fname = dir_name + '/potential_parameters.txt'
    fid = open(fname, 'w')
    fid.write('Well depth, De = ' + str(potential['Depth']) + '\n')
    fid.write('Well displacment, re = ' + str(potential['Distance']) + '\n')
    fid.write('Well width, a = ' + str(potential['Width']) + '\n')
    fid.close()


def save_inital_condistion(conditions, dir_name):
    fname = dir_name + '/initial_conditions.csv'
    fid = open(fname, 'w')
    fid.write('Number of atoms = {}\n'.format(conditions['n atom']))
    fid.write('Time step = {}\n'.format(conditions['Time step']))
    fid.write('Number of iterations = {}\n'.format(conditions['Iterations']))
    fid.write('x,y,v_x,v_y\n')
    for pos, v in zip(conditions['Position'], conditions['Velocity']):
        fid.write('{},{},{},{}\n'.format(pos[0], pos[1], v[0], v[1]))
    fid.close()


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
