# atom-potential-scatter

A library for following the classical trajectories of atoms in a Morse potential off surfaces.
The main intention is to scatter from rough surfaces and code is provided for running those
investigations, however any surface can be provided as the potential is calculated by interpolating
a list of surface points.

Rust code shold be compiled into a shared library with `cargo build --release` and then the python
module `atom_potential_scatter.py` can be used to interface with it in order to run simulations.
The script `run_scatter.py` is a higher level script that contains some functions that run
specific simulations.

## Python modules

For the interface module: `numpy`, `matplotlib`, `ctypes`

In addition the `run_scatter.py` script uses `pandas`.

## Rust crates

* `splines` for interpolation of the surfaces
* `nalgebra` for linear algebra operations
* `rayon` for parallel execution
