use std::env;
//use std::fs;

use atom_potential_scatter::RandSurface;

fn main() {
    let args: Vec<String> = env::args().collect();
    assert!(args.len() == 2);
    let n: usize = args[1].parse().unwrap();
    assert!(n >= 2);
    // Lets generate a simpdle surface
    let p: [f64; 3] = [1.0, 1.0, 1.0];
    let test_surface = RandSurface::generate_surface(5.0, 0.1, 10.0, n, p);
    test_surface.save_to_file("surf_gen_in_rust.csv");
    println!("I saved a random surface to file!");
}