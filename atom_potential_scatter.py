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
import csv
import matplotlib.pyplot as plt
import matplotlib.cm as cm

atom_scatter = ctypes.CDLL(("atom-potential-scatter/target/release/libatom_pot"
                            "ential_scatter.so"))


# --------------------------- Helper functions ------------------------------ #

def _parse_eq(s, typ='float'):
    if typ != 'float' and typ != 'int':
        raise ValueError('Only float and int types supported')
    num = float(s.split("=")[1]) if typ == 'float' else int(s.split("=")[1])
    return(num)


def _load_text(fname):
    with open(fname) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    return(content)


def _simulation_dir(name="just_a_test", nocreate=False):
    """Creates a name for a directory for the simultion being performed.
    Automatically numbers the directory with a 4 digit number and then creates
    that directory in 'results' and returns the full path."""

    if name[0] == "/":
        name = name[1:]
    if not os.path.isdir("results"):
        os.mkdir("results")
    past = list(map(lambda x: "results/" + x, os.listdir("results")))
    past_dirs = list(filter(os.path.isdir, past))
    inds = list(map(int, list(map(lambda s: s[8:12], past_dirs))))
    ind = 1 if len(past_dirs) == 0 else max(inds) + 1
    dir_name = "results/" + str(ind).zfill(4) + '-' + name
    if not os.path.isdir(dir_name) and not nocreate:
        os.mkdir(dir_name)
    return(dir_name)


def _format_num(num, frmt='short'):
    s = "{0:.4f}".format(num) if frmt == 'long' else "{0:.2f}".format(num)
    n = 7 if frmt == 'long' else 5
    if s[0] != '-':
        s = ' ' + s
    if len(s) > n:
        s = s[:n]
    elif len(s) <= n:
        s = ' '*(len(s)-n+1) + s
    return(s)


def _myround(x, direction="up", base=5):
    tmp = np.ceil(x/base) if direction == "up" else np.floor(x/base)
    return base*tmp


# --------------------------- Other small functions ------------------------- #

def morse_1d(ys, potential):
    """Calculates the morse potential for the given parameters and y
    position(s)"""

    de = potential['Depth']
    a = potential['Width']
    re = potential['Distance']
    return(de*(np.exp(-2.0*a*(ys - re)) - 2.0*np.exp(-a*(ys - re))))


def ye_from_De(De, a, he_energy=1.0):
    return((1/a)*np.log(1 + np.sqrt(1 + 1/De)))


# TODO: choose a value for the constant
def choose_time_step(pot, const):
    """From the parameters of a potential a timesstep is chosen such that the
    product of the maximum gradient and the timestep are constant."""
    e = pot.Width*pot.Displacement
    Dt = const/(pot.Depth*pot.Width*( np.exp(e) - np.exp(2*e) ))
    return(Dt)

# --------------------------- Classes --------------------------------------- #

class Surf:
    def __init__(self):
        self.Dx = 0.0
        self._x = np.array([])
        self._f = np.array([])
        self.N_points = 0
        self.__type = 'debug'

    def __repr__(self):
        s = "Number of points in surface: {}\n".format(self.N_points)
        s = s + "Point seperation: {}\n".format(self.Dx)
        s = s + "Type of surface: {}\n".format(self.__type)
        s = s + "First 5 points of the surface:\n      x,       y\n"
        for i, (x, y) in enumerate(zip(self._x, self._f)):
            s1 = _format_num(x, frmt='long')
            s2 = _format_num(y, frmt='long')
            s = s + s1[1:] + ',' + s2 + '\n'
            if i == 4:
                s = s + '...'
                break
        return(s)

    def set_surface(self, x, f, Dx):
        if len(x) != len(f):
            raise ValueError('x and y lengths are not the same')
        if sum((abs(np.diff(x) - Dx)) < 1e-10) != len(x) - 1:
            raise ValueError(('The given x values do not all have the '
                              'expected spacing.'))
        self._x = x
        self._f = f
        self.N_points = len(x)
        self.Dx = Dx
        self.__type = 'interpolate'

    def get_points(self):
        return(self._x, self._f)

    def get_type(self):
        return(self.__type)

    def as_dict(self):
        return({'x': self._x, 'y': self._f, 'type': 'interpolate'})


DEFAULT_SURF = Surf()


class RandSurf(Surf):
    def __init__(self):
        super().__init__()
        self.lambd = 0.0
        self.h = 0.0
        self.h_RMS = 0.0
        self.corr_len = 0.0

    def __repr__(self):
        s = "Number of points in surface: {}\n".format(self.N_points)
        s = s + "Point seperation: {}\n".format(self.Dx)
        s = s + "RMS height: {}\n".format(self.h_RMS)
        s = s + "Correlation length: {}\n".format(self.corr_len)
        s = s + "First 5 points of the surface:\n      x,       y\n"
        for i, (x, y) in enumerate(zip(self._x, self._f)):
            s1 = _format_num(x, frmt='long')
            s2 = _format_num(y, frmt='long')
            s = s + s1[1:] + ',' + s2 + '\n'
            if i == 4:
                s = s + '...'
                break
        return(s)

    def random_surf_gen(h_RMS, Dx, corr_len, N=10001):
        self = RandSurf()
        if corr_len/Dx < 10:
            raise ValueError(("The segment size Dx is not sufficently small "
                              "compared to the correlation length"))
        self.lambd = 0.5*corr_len**(2/3)
        self.h = h_RMS*np.sqrt(np.tanh(Dx/self.lambd))
        f, x = RandSurf.__random_surf_gen_core(self.h, Dx, self.lambd, 1, N)
        self.set_surface(x, f, Dx)
        self.h_RMS = h_RMS
        self.corr_len = corr_len
        return(self)

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
        ax.plot(self._x, self._f)
        ax.set_title('Generated surface profile')
        ax.set_xlabel('x/nm')
        ax.set_ylabel('y/nm')
        lims = [-self.Dx*5000, self.Dx*5000] if self.N_points > 10000 \
            else np.array([-1, 1])*self.Dx*round(self.N_points/2)
        ax.set_xlim(lims)

        fig2 = plt.figure()
        ax2 = fig2.add_axes([0, 0, 0.6, 0.6])
        ax2.hist(self._f, density=True, label='Generated')
        end = max(abs(self._f))
        xx = np.linspace(-end, end, 501)
        yy = 1/(np.sqrt(2*np.pi)*self.h_RMS) * np.exp(-xx**2/(2*self.h_RMS**2))
        ax2.plot(xx, yy, label='Predicted')
        ax2.set_xlabel('Heights/nm')
        ax2.set_ylabel('Probability')
        ax2.set_title('Height distribution')
        ax2.legend(loc='lower right')

        fig3 = plt.figure()
        ax3 = fig3.add_axes([0, 0, 0.6, 0.6])
        auto_corr = np.correlate(self._f, self._f, mode='same')
        auto_corr = auto_corr/max(auto_corr)
        corr_x = np.arange(-len(auto_corr)*self.Dx/2, len(auto_corr)*self.Dx/2,
                           self.Dx)
        ax3.plot(corr_x, auto_corr, label='Generated')
        ax3.plot(corr_x, np.exp(-abs(corr_x)/self.lambd) *
                 (1 + abs(corr_x)/self.lambd), label='Predicted')
        ax3.set_xlabel('x/nm')
        ax3.set_ylabel('Correleation')
        ax3.set_xlim(0, 2*self.corr_len)
        ax3.set_ylim(0, 1)
        ax3.set_title('Autocorrelation function')
        ax3.legend()

        fig4 = plt.figure()
        ax4 = fig4.add_axes([0, 0, 0.6, 0.6])
        grad = np.diff(self._f)/self.Dx
        ax4.hist(grad, density=True, label='Generated')
        end = max(abs(grad))
        xx = np.linspace(-end, end, 501)
        yy = (self.lambd/(np.sqrt(2*np.pi)*self.h_RMS)) * \
            np.exp(-self.lambd**2 * xx**2/(2*self.h_RMS**2))
        ax4.plot(xx, yy, label='Predicted')
        ax4.set_xlabel('Gradients')
        ax4.set_ylabel('Probability')
        ax4.set_title('Gradient distribution')
        ax4.legend(loc='lower right')

        return([(fig, ax), (fig2, ax2), (fig3, ax3), (fig4, ax4)])

    def plot_potential(self, param):
        fig1 = plt.figure()
        ax1 = fig1.add_axes([0, 0, 1.2, 0.5])
        ax1.plot(self._x, self._f)
        ax1.set_xlim([-self.Dx*1000, self.Dx*1000])
        ax1.set_xlabel('x/nm')
        ax1.set_ylabel('y/nm')

        xx = np.linspace(-80, 80, 401)
        yy = np.linspace(-5, 30, 171)
        g = np.meshgrid(xx, yy)
        V = morse_potential(g[0].flatten(), g[1].flatten(), param,
                            self).reshape((171, 401))
        fig2 = plt.figure()
        ax2 = fig2.add_axes([0, 0, 1.2, 0.5])
        cs = ax2.contourf(g[0], g[1], V, cmap=cm.coolwarm,
                          levels=np.linspace(-1, 1, 21), extend='max')
        ax2.set_xlabel("x/nm")
        ax2.set_ylabel("y/nm")
        fig2.colorbar(cs, ax=ax2)
        ax2.set_xlim([-self.Dx*1000, self.Dx*1000])
        ax2.set_ylim([-param.Displacemnet*4, param.Width*10])
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
        for x, y in zip(self._x, self._f):
            fid.write('{},{}\n'.format(x, y))
        fid.close()

    def load_surf(dir_name):
        fname = dir_name + '/surface_used.csv'
        content = _load_text(fname)
        self = RandSurf()
        self.h_RMS = _parse_eq(content[1])
        self.corr_len = _parse_eq(content[2])
        self.h = _parse_eq(content[4])
        Dx = _parse_eq(content[5])
        self.lambd = _parse_eq(content[6])
        # Take the remainder of the lines and turn them into two lists of x,y
        x = np.zeros(len(content) - 9)
        f = np.zeros(len(content) - 9)
        for i, c in enumerate(content[9:]):
            x[i], f[i] = tuple(map(float, c.split(",")))
        self.set_surface(x, f, Dx)
        return(self)


class Potential:
    """Contains the three parameters for the """
    def __init__(self, De=0.1, re=0.0, a=1.0):
        self.Depth = De
        self.Displacement = re
        self.Width = a

    def __repr__(self):
        return("Depth: {}\nDisplacment: {}nm\nWidth: {}nm".format(self.Depth,
               self.Displacement, self.Width))

    def as_dict(self):
        return({'Depth': self.Depth, 'Width': self.Width, 'Displacement':
                self.Dusplacement})

    def save_potential(self, dir_name):
        fname = dir_name + '/potential_parameters.txt'
        fid = open(fname, 'w')
        fid.write('Well depth, De = ' + str(self.Depth) + '\n')
        fid.write('Well displacment, re = ' + str(self.Displacement) + '\n')
        fid.write('Well width, a = ' + str(self.Width) + '\n')
        fid.close()

    def load_potential(dir_name):
        self = Potential()
        fname = dir_name + '/potential_parameters.txt'
        content = _load_text(fname)
        self.Depth = _parse_eq(content[0])
        self.Displacement = _parse_eq(content[1])
        self.Width = _parse_eq(content[2])
        return(self)


class Conditions:
    def __init__(self, n_atom=0, Dt=0.05, n_it=1000):
        self.n_atom = n_atom
        self.Dt = Dt
        self.n_it = int(n_it)
        self.position = np.zeros([2, self.n_atom])
        self.velocity = np.zeros([2, self.n_atom])

    def __repr__(self):
        s = "Number of atoms: {}\nTime step: {}\n".format(self.n_atom, self.Dt)
        s = s + "Number of iterations: {}\n".format(self.n_it)
        t1 = ("Initial conditions of the first 5 atoms:\n    x,     y,    vx,"
              "    vy\n")
        t2 = "Initial conditions of the atoms:\n    x,     y,    vx,    vy\n"
        s = s + t1 if self.n_atom > 5 else s + t2
        for i, (p, v) in enumerate(zip(self.position.transpose(),
                                       self.velocity.transpose())):
            s1 = _format_num(p[0])
            s2 = _format_num(p[1])
            s3 = _format_num(v[0])
            s4 = _format_num(v[1])
            s = s + s1[1:] + ',' + s2 + ',' + s3 + ',' + s4 + '\n'
            if i == 4:
                s = s + '...'
                break
        return(s)

    def set_velocity(self, init_angle, speed):
        v = speed*np.array([[np.sin(init_angle*np.pi/180)],
                           [-np.cos(init_angle*np.pi/180)]])
        self.velocity = np.repeat(v, self.n_atom, axis=1)

    def set_position(self, x_range, y):
        xs = np.linspace(x_range[0], x_range[1], self.n_atom)
        ys = np.repeat(y, self.n_atom)
        self.position = np.array([xs, ys])

    def save_inital_conditions(self, dir_name):
        fname = dir_name + '/initial_conditions.csv'
        fid = open(fname, 'w')
        fid.write('Number of atoms = {}\n'.format(self.n_atom))
        fid.write('Time step = {}\n'.format(self.Dt))
        fid.write('Number of iterations = {}\n'.format(self.n_it))
        fid.write('x,y,v_x,v_y\n')
        for p, v in zip(self.position.transpose(), self.velocity.transpose()):
            fid.write('{},{},{},{}\n'.format(p[0], p[1], v[0], v[1]))
        fid.close()

    def load_initial_conditions(dir_name):
        fname = dir_name + '/initial_conditions.csv'
        content = _load_text(fname)
        n_atom = _parse_eq(content[0], typ='int')
        dt = _parse_eq(content[1])
        n_it = _parse_eq(content[2])
        self = Conditions(n_atom, dt, n_it)
        for i, c in enumerate(content[4:]):
            self.position[0][i], self.position[1][i], self.velocity[0][i], \
                self.velocity[1][i] = tuple(map(float, c.split(",")))
        return(self)


class Trajectory:
    def __init__(self, n):
        self.position = np.zeros([2, n])*float('nan')
        self.velocity = np.zeros([2, n])*float('nan')
        self.times = np.zeros(n)
        self.Dt = 1.0
        self.n_skip = 1
        self.pos_error = np.zeros([2, n])*float('nan')
        self.vel_error = np.zeros([2, n])*float('nan')
        self.records = n

    def __repr__(self):
        s = "Number of records: {}\nTime step: {}\n".format(self.records,
                                                            self.Dt)
        s = s + "1 in x recorded x: {}\n".format(self.n_skip)
        t1 = ("First 5 points in the trajectory:\n    t,     x,     y,    vx,"
              "    vy\n")
        t2 = "Trajectory of the atom:\n    t,     x,     y,    vx,    vy\n"
        if not np.isnan(self.pos_error[0][0]):
            s = s + "Error values present\n"
        s = s + t1 if self.records > 5 else s + t2
        for i, (t, p, v) in enumerate(zip(self.times,
                                          self.position.transpose(),
                                          self.velocity.transpose())):
            s0 = _format_num(t)
            s1 = _format_num(p[0])
            s2 = _format_num(p[1])
            s3 = _format_num(v[0])
            s4 = _format_num(v[1])
            s = s + s0[1:] + ',' + s1 + ',' + s2 + ',' + s3 + ',' + s4 + '\n'
            if i == 4:
                s = s + '...'
                break
        return(s)

    def load_trajectory(dir_name, n_skip=10):
        d = "results/" + dir_name if dir_name[0:7] != "results" else dir_name
        with open(d) as csv_file:
            csv_reader = csv.DictReader(csv_file, delimiter=',')
            count = sum(1 for _ in csv_reader)
            csv_file.seek(0)
            csv_reader = csv.DictReader(csv_file, delimiter=',')
            traj = Trajectory(count)
            for i, row in enumerate(csv_reader):
                traj.times[i] = row['time']
                traj.position[0][i] = row['x']
                traj.position[1][i] = row['y']
                traj.velocity[0][i] = row['vx']
                traj.velocity[1][i] = row['vy']
                if "e_x" in row:
                    traj.pos_error[0][i] = row['e_x']
                    traj.pos_error[1][i] = row['e_y']
                    traj.vel_error[0][i] = row['e_vx']
                    traj.vel_error[1][i] = row['e_vy']
        traj.n_skip = n_skip
        traj.Dt = (traj.times[2] - traj.times[1])/n_skip
        return(traj)

    def energy(self, surf, potential, mass=1.0):
        kin_energy = 0.5*mass*np.sum(self.velocity**2, 0)
        pot_energy = morse_potential(self.position[0], self.position[1],
                                     potential, surf)
        return(kin_energy, pot_energy)

    def energy_deviation(self, surf, potential, mass=1.0):
        kin_energy, pot_energy = self.energy(surf, potential)
        init_energy = kin_energy[0] + pot_energy[0]
        deviation = 100*(kin_energy + pot_energy)/init_energy - 100
        return(deviation)

    def conservation_of_energy(self, surf, potential, mass=1.0):
        kin_energy, pot_energy = self.energy(surf, potential)
        fig = plt.figure()
        ax = fig.add_axes([0, 0, 1.0, 0.7])
        ax.plot(self.times, kin_energy, label='Kinetic energy')
        ax.plot(self.times, pot_energy, label='Potential energy')
        ax.plot(self.times, kin_energy + pot_energy, label='Total energy')
        ax.legend()
        ax.set_xlabel('Time')
        ax.set_ylabel('Energy')
        return(fig, ax)

    def plot_traj(self, surf, potential, fig=None, ax=None):
        if ax is None:
            fig, ax = plot_potential_2d(surf, potential,
                                        figsize=[0, 0, 1.0, 0.6])
        ax.plot(self.position[0], self.position[1], color='black')
        m1 = _myround(min(self.position[0]), direction="down")
        m2 = _myround(max(self.position[0]), direction="up")
        ax.set_xlim(m1, m2)
        ax.set_ylim(-5, _myround(max(self.position[1])))
        return(fig, ax)

    def error_plot(self, fig=None, ax=None):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_axes([0, 0, 1.0, 0.6])
        ax.plot(self.times, self.pos_error[0], color='blue', label='x')
        ax.plot(self.times, self.pos_error[1], color='red', label='y')
        ax.plot(self.times, self.vel_error[0], color='green', label='vx')
        ax.plot(self.times, self.vel_error[1], color='black', label='vy')
        ax.legend()
        return(fig, ax)

    def cuml_error(self):
        return({'x': np.sum(abs(self.pos_error[0])),
                'y': np.sum(abs(self.pos_error[1])),
                'vx': np.sum(abs(self.vel_error[0])),
                'vy': np.sum(abs(self.vel_error[1]))})


class FinalStates:
    def __init__(self, n):
        self.position = np.zeros([2, n])*float('nan')
        self.velocity = np.zeros([2, n])*float('nan')
        self.n_atom = n
        self.n_bound = 0
        self.beyond_surf = 0
        self.state_of_atom = np.array(['']*n)

    def __repr(self):
        print("print the final states")

    def load_states(dir_name, surf, potential):
        d = "results/" + dir_name if dir_name[0:7] != "results" else dir_name
        with open(d) as csv_file:
            csv_reader = csv.DictReader(csv_file, delimiter=',')
            count = sum(1 for _ in csv_reader)
            csv_file.seek(0)
            csv_reader = csv.DictReader(csv_file, delimiter=',')
            states = FinalStates(count)
            for i, row in enumerate(csv_reader):
                states.position[0][i] = row['x']
                states.position[1][i] = row['y']
                states.velocity[0][i] = row['vx']
                states.velocity[1][i] = row['vy']
                states.state_of_atom[i] = 'beyond' if np.isnan(row['x']) \
                    else 'ok'
        states.beyond_surf = np.sum(states.state_of_atom == 'beyond')
        states.filter_escaped(surf, potential)
        return(states)

    def filter_escaped(self, surf, potential):
        pt = morse_potential(self.position[0], self.position[1], potential,
                             surf)
        ke_y = 0.5*self.velocity[1]**2
        localised = ke_y > -pt
        self.state_of_atom[localised] = 'bound'
        self.n_bound = sum(localised)

    def get_escaped(self):
        ind = self.state_of_atom == 'ok'
        pos = self.position[ind]
        vel = self.velocity[ind]
        return(pos, vel)

    def final_direction_plot(self):
        _, vel = self.get_escaped()
        theta = np.arctan(vel[0]/vel[1])*180/np.pi
        fig = plt.figure()
        ax = fig.add_axes([0, 0, 0.7, 0.7])
        ax.hist(theta, bins=36, density=True)
        ax.set_xlim([-90, 90])
        ax.set_xlabel(r'$\theta/^o$')
        ax.set_ylabel('Probability')
        return(fig, ax)


# --------------------------- Display functions ----------------------------- #

def plot_potential_2d(surf, potential, ax=None, figsize=[0, 0, 2, 0.5]):
    xx = np.linspace(-80, 80, 401)
    yy = np.linspace(-5, 30, 171)
    g = np.meshgrid(xx, yy)
    V = morse_potential(g[0].flatten(), g[1].flatten(), potential,
                        surf).reshape((171, 401))
    if ax is None:
        fig = plt.figure()
        ax = fig.add_axes(figsize)
    cs = ax.contourf(g[0], g[1], V, cmap=cm.coolwarm,
                     levels=np.linspace(-1, 1, 21), extend='max')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(cs, ax=ax)
    ax.set_title("Atom potential")
    return(fig, ax)


# def trajectory_plot(dir_name, surf, potential):
#    d = "results/" + dir_name if dir_name[0:7] != "results" else dir_name
#    # Plot the trajectories
#    (fig, ax) = plot_potential_2d(surf, potential)
#    fnames = os.listdir(path=d)
#    fnames2 = list(filter(lambda k: "atom" in k, fnames))
#    for f in fnames2:
#        t = Trajectory.load_trajectory(f)
#        t.plot_traj(ax=ax)
#    ax.set_ylim([-1, 20])
#    ax.set_xlim([-40, 40])
#    ax.set_title("Atom potential and trajectory")
#    fig.savefig(d + "/many_trajectories_rust.pdf", bbox_inches="tight")
#

# --------------------------- Interface functions --------------------------- #

def run_single_particle(init_pos, init_v, dt, it, sim_name, potential,
                        surf=DEFAULT_SURF, method="Fehlberg", new_dir=True):
    """Runs a single particle with the specified initial conditions through the
    potential for a given number of iteration with the given timestep."""

    # Put the surface information into the correct types
    xs, ys = surf.get_points()
    t = surf.get_type() == 'interpolate'
    surf_x = xs.ctypes.data_as(POINTER(c_double)) if t \
        else (c_double*0)(*[])
    surf_y = ys.ctypes.data_as(POINTER(c_double)) if t \
        else (c_double*0)(*[])
    test_surf = c_uint64(0) if t else c_uint64(1)
    surf_n = c_uint64(surf.N_points) if t else c_uint64(0)
    # Put the potential values into a C array of doubles
    p = (c_double*3)(*[potential.Depth, potential.Displacement,
                       potential.Width])
    # Put the initial conditions into C arrays of doubles
    x = c_double(init_pos[0])
    y = c_double(init_pos[1])
    vx = c_double(init_v[0])
    vy = c_double(init_v[1])

    # Put the directory name and integration method into a C array of chars
    d = _simulation_dir(sim_name) + '/atom_traajectory.csv' if new_dir else sim_name
    arr = (c_char*len(d))(*d.encode('ascii'))
    mth = (c_char*len(method))(*method.encode('ascii'))
    atom_scatter.single_particle(x, y, vx, vy, c_double(dt), c_uint64(it), arr,
                                 c_uint64(len(d)), p, mth, c_uint64(len(mth)),
                                 surf_x, surf_y, surf_n, test_surf, c_uint64(1))
    return(d)


def run_many_particle(init_cond, sim_name, potential, record,
                      surf=DEFAULT_SURF, method="Fehlberg"):
    # Put the surface information into the correct types
    t = surf.get_type() == 'interpolate'
    x, y = surf.get_points()
    surf_x = x.ctypes.data_as(POINTER(c_double)) if t else (c_double*0)(*[])
    surf_y = y.ctypes.data_as(POINTER(c_double)) if t else (c_double*0)(*[])
    test_surf = c_uint64(0) if t else c_uint64(1)
    surf_n = c_uint64(len(surf._x)) if t else c_uint64(0)

    # Put the potential values into a C array of doubles
    p = (c_double*3)(*[potential.Depth, potential.Displacement,
                       potential.Width])
    # Put the initial conditions into C arrays of doubles
    xs = init_cond.position[0, ].ctypes.data_as(POINTER(c_double))
    ys = init_cond.position[1, ].ctypes.data_as(POINTER(c_double))
    vxs = init_cond.velocity[0, ].ctypes.data_as(POINTER(c_double))
    vys = init_cond.velocity[1, ].ctypes.data_as(POINTER(c_double))

    # Put the directory name and integration method into a C array of chars
    d = _simulation_dir(sim_name)
    arr = (c_char*len(d))(*d.encode('ascii'))
    meth = (c_char*len(method))(*method.encode('ascii'))

    # Number of atoms we are simulating as a C int
    n_atom = c_uint64(np.shape(init_cond.position)[1])
    atom_scatter.multiple_particle(xs, ys, vxs, vys, n_atom, c_double(init_cond.Dt),
                                   c_uint64(init_cond.n_it), arr, c_uint64(len(d)), p,
                                   c_uint64(record), meth, c_uint64(len(meth)),
                                   surf_x, surf_y, surf_n, test_surf, c_uint64(10))
    return(d)


def morse_potential(xs, ys, potential, surf=DEFAULT_SURF):
    """Uses the implementation of the morse potential functioin in rust to
    evaluate it for the given numpy arrays of x,y."""

    # Put the surface information into the correct types
    t = surf.get_type() == 'interpolate'
    x, y = surf.get_points()
    surf_x = x.ctypes.data_as(POINTER(c_double)) if t else (c_double*0)(*[])
    surf_y = y.ctypes.data_as(POINTER(c_double)) if t else (c_double*0)(*[])
    test_surf = c_uint64(0) if t else c_uint64(1)
    surf_n = c_uint64(surf.N_points) if t else c_uint64(0)

    p = (c_double*3)(*[potential.Depth, potential.Displacement,
                       potential.Width])
    n = xs.shape[0]
    i = xs.ctypes.data_as(POINTER(c_double))
    j = ys.ctypes.data_as(POINTER(c_double))
    V = np.zeros(n)
    k = V.ctypes.data_as(POINTER(c_double))
    atom_scatter.calc_potential(i, j, k, c_uint64(n), p, surf_x, surf_y,
                                surf_n, test_surf)
    return(V)


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


def interp_test(x, y, xs):
    i = x.ctypes.data_as(ctypes.POINTER(c_double))
    j = y.ctypes.data_as(ctypes.POINTER(c_double))
    k = xs.ctypes.data_as(ctypes.POINTER(c_double))
    ys = np.zeros(xs.shape[0])
    ll = ys.ctypes.data_as(ctypes.POINTER(c_double))
    atom_scatter.interpolate_test(i, j, c_uint64(x.shape[0]), k, ll,
                                  c_uint64(xs.shape[0]))
    return(ys)
