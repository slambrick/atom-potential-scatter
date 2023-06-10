use std::env;
extern crate rand;
//use std::fs;

use rand_distr::{Normal, Distribution};
use rand::prelude::*;

use atom_potential_scatter::RandSurface;

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