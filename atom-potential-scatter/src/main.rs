use std::env;
extern crate rand;
extern crate nalgebra as na;
//use std::fs;

use na::{Vector4, Matrix4, Vector2};
use rand_distr::{Normal, Distribution};
use rand::prelude::*;
use atom_potential_scatter::{RandSurface, Potential, run_particle, SimParam, Atom};

// Mass of the helium atom is set to 1
// Units of distance are in nm
// Set the kinetic energy of the atoms to be 1 for 300K He-4 atoms
// Therefore the speed of the atoms is sqrt(2) for 300K He-4 atoms
// This sets the arbitrary units of time: 8.0116735e-13 units/s
// Double check the above...



fn main() {
    let args: Vec<String> = env::args().collect();
    assert!(args.len() == 2);
    let n: usize = args[1].parse().unwrap();
    assert!(n >= 2);
    // Lets generate a simpdle surface
    let p: [f64; 3] = [1.0, 1.0, 1.0];
    
    test_number_gen(n);
    
    let test_surface = RandSurface::generate_surface(5.0, 0.1, 10.0, n, p);
    test_surface.save_to_file("surf_gen_in_rust.csv");
    println!("I saved a random surface to file!");
}


/// Loads in simulation parameters from the parameters file.
fn load_param() {
    // TODO
}


/// Runs a single particle through a potential, the test gaussian bump is used here.
fn single_particle_test() {

    //load_param(); //TODO: this will load all the parameters at some point

    // Parameters for the integration
    let n_it = 10000;       // Max number of iterations to stop after
    let height_stop: f64 = 10.0; // The height above the surface to stop the simulation
    let h = 0.0; // TODO: what is h?
    let skip_record = 5;   // Record every 5th timestep
    let method = "Verlet"; // Verlet integration (there are a few methods implemented)

    let h_rms: f64 = 0.5; // The RMS height for the generated surface (nm)
    let corr: f64 = 10.0; // The correlation length for the generated surface (nm)
    let dx: f64 = 0.05;   // The element length in the surface (nm)
    let p: [f64; 3] = [h_rms, corr, dx];

    let pot = Potential::gauss_potential(p);
    let init_q = Vector4::new(0.0, 0.0, 0.0, 0.0);
    let sim_params = SimParam{
        skip_record: skip_record as usize,
        h,
        param: pot,
        n_it: n_it as usize,
        method: method.to_string(),
        height_stop
    };

    // Actually run a simulation on an atom
    let he_atom: Atom = run_particle(init_q, &sim_params, true);
    let save_fname = "test_trajectory.csv";
    he_atom.write_trajectory(&save_fname, h);
}


/// This function quickly tests the number generation of 
fn test_number_gen(n : usize) {
    let dx = 0.1;
    let lambda = 3.0;
    let s = 5.0;

    // Random Gaussian points
    let mut zs: Vec<f64> = vec![0.0; n];
    let normal = Normal::new(0.0, s).unwrap();
    let mut rng: ThreadRng = rand::thread_rng();
    let mut n_rand = normal.sample_iter(&mut rng);
    for i in 0..n  {
        zs[i] = n_rand.next().unwrap();
    }
    
    // Generate exponential points to represent the correlation length
    let mut ms: Vec<f64> = vec![0.0; n];
    let mut es: Vec<f64> = vec![0.0; n];
    let mut xs: Vec<f64> = vec![0.0; n];
    for i in 0..n {
        ms[i] = -(n as f64)/2.0 + (i as f64);
        es[i] = (- ms[i].abs() * dx/lambda).exp();
        xs[i] = ms[i]*dx;
    }
}