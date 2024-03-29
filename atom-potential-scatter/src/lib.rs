#![crate_name = "atom_potential_scatter"]

extern crate nalgebra as na;
extern crate rayon;
extern crate rand_distr;
extern crate rand;
extern crate fftconvolve;
extern crate ndarray;

use na::{Vector4, Matrix4, Vector2};
use core::f64::consts::PI;
use std::slice;
use std::fs::OpenOptions;
use std::io::prelude::*;
use std::str;
use std::convert::AsMut;
use splines::{Interpolation, Key, Spline};
use rayon::prelude::*;
use rand_distr::{Normal, Distribution};
use rand::prelude::*;
use fftconvolve::{fftconvolve, Mode};
use ndarray::{Array, ArrayBase, Axis, Data, Dimension, Slice, Array1}; 
// TODO: I should generally be using ndarray for data stuff

/// Constans used by the R-K-F integration method.
/// These value fill up the Butcher tableau
const A21: f64 = 1.0/4.0;
const A31: f64 = 3.0/32.0;
const A32: f64 = 9.0/32.0;
const A41: f64 = 1932.0/2197.0;
const A42: f64 = -7200.0/2197.0;
const A43: f64 = 7296.0/2197.0;
const A51: f64 = 439.0/216.0;
const A52: f64 = -8.0;
const A53: f64 = 3680.0/513.0;
const A54: f64 = -845.0/4104.0;
const A61: f64 = -8.0/27.0;
const A62: f64 = 2.0;
const A63: f64 = -3544.0/2565.0;
const A64: f64 = 1859.0/4104.0;
const A65: f64 = -11.0/40.0;

const B1: f64 = 16.0/135.0;
const B3: f64 = 6656.0/12825.0;
const B4: f64 = 28561.0/56430.0;
const B5: f64 = -9.0/50.0;
const B6: f64 = 2.0/55.0;

const B1_: f64 = 25.0/216.0;
const B3_: f64 = 1408.0/2565.0;
const B4_: f64 = 2197.0/4104.0;
const B5_: f64 = -1.0/5.0;

/// Performs a 1D convolution for two arrays of the same length, using the naive
/// approach. Always reduces the length of the output to the length of the input
/// and assumes that the two vectors are the same length.
/// 
/// * `f` - first vector, length n
/// * `g` - second vector, length n
/// 
fn convolve(f: Vec<f64>, g: Vec<f64>) -> Vec<f64> {
    assert!(f.len() == g.len());
    assert!(f.len() >= 2);
    let n: usize = f.len();
    let len_output: usize = 2*n - 1;
    let mut full_output = vec![0.0; len_output];
    let mut output = vec![0.0; n];

    for i in 0..len_output {
        let mut val = 0.0;
        for j in 0..g.len() {
            if i >= j && i - j < n {
                val += f[j] * g[i - j];
            }
        }
        full_output[i] = val;
    }

    let skip: usize = (len_output - n)/2;
    for i in 0..n {
        output[i] = full_output[i + skip];
    }

    output
}

/// Faster convolution using FFT, only for 1D arrays.
/// 
/// Why? Because I want to write my own, that's why.
fn convolve_fft(f: Vec<f64>, g: Vec<f64>)  {
    // TODO: do some checking
    assert!(f.len() == g.len());
    let n = f.len();

    // Pad the arrays to the next power of 2
    
    // Take the fft of both and multiply them to get the convolution

    // Check the length of the output
}

/// Copys a slice into an array
fn copy_into_array<A, T>(slice: &[T]) -> A
where
    A: Default + AsMut<[T]>,
    T: Copy,
{
    let mut a = A::default();
    <A as AsMut<[T]>>::as_mut(&mut a).copy_from_slice(slice);
    a
}

/// Takes two vectors of x and y positions and applies cubic hermite polynomial
/// interpolation to those to create a Spline for evaluating the values of
/// surface heights. Returns the spline.
fn interpolate_surf(xs: &Vec<f64>, ys: &Vec<f64>) -> Spline<f64,f64> {
    assert!(xs.len() == ys.len());
    let mut ks = vec![Key::new(0.0, 0.0, Interpolation::CatmullRom); xs.len()];
    for i in 0..xs.len() {
        ks[i] = Key::new(xs[i], ys[i], Interpolation::CatmullRom);
    }
    Spline::from_vec(ks)
}

/// A simple gaussian function to use as a test surface. The mean is always 0.
///
/// # Arguments
///
/// * `x` - The value to calculate the Gaussian for
/// * `s` - Standard deviation
/// * `height` - Vertical scaling of the Gaussian
fn gaussian(x: f64, s: f64, height: f64) -> f64{
    height*(1.0/(s*(2.0*PI).sqrt()))*(-x.powi(2)/(2.0*s.powi(2))).exp()
}


fn morse_gauss(x: f64, y: f64, p: &Potential) -> f64 {
    let r = y - gaussian(x, 5.0, 30.0);
    p.de*( (-2.0*p.a*(r - p.re)).exp() - 2.0*(-p.a*(r - p.re)).exp() )
}

fn diff_at_point_gauss(x: f64) -> f64 {
    let epsilon: f64 = 0.01;
    (gaussian(x + 0.5*epsilon, 5.0, 30.0) - gaussian(x - 0.5*epsilon, 5.0, 30.0))/epsilon
}

/// Calculates the value of the potential for a given (x,y) position, for the
/// specified potential field.
///
/// # Arguments
///
/// * `x`   - x position to evauate at
/// * `y`   - y position to evauate at
/// * `p`   - Parameters of the potential
/// * `spl` - Spline of the surface
fn morse(x: f64, y: f64, p: &Potential) -> f64 {
    let y_dip = match p.surf.clamped_sample(x) {
        Some(val) => val,
        None      => f64::NAN
    };
    let r = y - y_dip;
    p.de*( (-2.0*p.a*(r - p.re)).exp() - 2.0*(-p.a*(r - p.re)).exp() )
}

/// Calculates the local derivative at a specified point on the provided spline
///
/// # Arguments
///
/// TODO: this
fn diff_at_point(x: f64, spl: &Spline<f64,f64>) -> f64 {
    let epsilon: f64 = 0.01;
    let dy = match spl.clamped_sample(x + 0.5*epsilon) {
        Some(val) => val,
        None      => f64::NAN
    } - match spl.clamped_sample(x - 0.5*epsilon) {
        Some(val) => val,
        None      => f64::NAN
    };
    dy/epsilon
}

/// Hmm, what does this do?
/// It is part of the Runge-Kutter methods
fn c(q: Vector4<f64>, p: &Potential) -> Vector4<f64> {
    let r = if p.gauss {
        q[1] - gaussian(q[0], 5.0, 30.0)
    } else {
        q[1] - match p.surf.clamped_sample(q[0]) {
            Some(val) => val,
            None      => f64::NAN
        }
    };
    let dd = if p.gauss {
        diff_at_point_gauss(q[0])
    } else {
        diff_at_point(q[0], &p.surf)
    };
    // Derivative of the potential gives us the acceleration if the mass is assumed to be 1
    let ax = 2.0*p.a*p.de*dd*( (-2.0*p.a*(r - p.re)).exp() - (-p.a*(r - p.re)).exp() );
    let ay = p.de*( -2.0*p.a*(-2.0*p.a*(r - p.re)).exp() + 2.0*p.a*(-p.a*(r - p.re)).exp() );
    Vector4::new(0.0, 0.0, -ax, -ay)
}

/// Performs a single iteration of the 'classic' Runge-Kutta method
/// 
/// It is probably beter to use the R-K-F method than this one.
/// 
/// # Arguments
/// * `q` - Initial state of the atom
/// * `h` - timestep parameter
/// * `param` - The potential we are moving in
fn runge_kutter_classic(q: Vector4<f64>, h: f64, param: &Potential) -> (Vector4<f64>, Vector4<f64>, Vector2<f64>) {
    let b = Matrix4::new(0.0, 0.0, 1.0, 0.0,
                         0.0, 0.0, 0.0, 1.0,
                         0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0);
    let k1 = b*q + c(q, param);
    let k2 = b*q + c(q + h*k1/2.0, param);
    let k3 = b*q + c(q + h*k2/2.0, param);
    let k4 = b*q + c(q + h*k3, param);
    let q_new = q + (1.0/6.0)*h*(k1 + 2.0*k2 + 2.0*k3 + k4);
    (q_new, q_new, acceleration(q, param))
}

/// Performs a single iteration of the 'Runge-Kutta-Fehlberg' method. Returns
/// a tuple containing the results from both the 5th and 4th order methods
/// 
/// In theory the RKF allows for dynamic calculation of the timestep, "adaptive
/// Runge-Kutta", TODO: implement that.
/// 
/// # Arguments
/// * `q` - Initial state of the atom
/// * `h` - timestep parameter
/// * `param` - The potential we are moving in
/// 
/// Returns a tupe of (q_5th, q_4th, acceleration)
fn runge_kutter_fehlberg(q: Vector4<f64>, h: f64, param: &Potential) -> (Vector4<f64>, Vector4<f64>, Vector2<f64>) {
    let b = Matrix4::new(0.0, 0.0, 1.0, 0.0,
                         0.0, 0.0, 0.0, 1.0,
                         0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0);
    let k1 = b*q + c(q, param);
    let k2 = b*q + c(q + h*(A21*k1), param);
    let k3 = b*q + c(q + h*(A31*k1 + A32*k2), param);
    let k4 = b*q + c(q + h*(A41*k1 + A42*k2 + A43*k3), param);
    let k5 = b*q + c(q + h*(A51*k1 + A52*k2 + A53*k3 + A54*k4), param);
    let k6 = b*q + c(q + h*(A61*k1 + A62*k2 + A63*k3 + A64*k4 + A65*k5), param);

    let q_5th = q + h*(k1*B1 + k3*B3 + k4*B4 + k5*B5 + k6*B6);
    let q_4th = q + h*(k1*B1_ + k3*B3_ + k4*B4_ + k5*B5_);
    (q_5th, q_4th, acceleration(q, param))
}

/// Calculates the relative step size for the next iteration based on the
/// error from the previous step.
fn new_step_size(h_old: f64, epsilon: f64, q5: &Vector4<f64>, q4: &Vector4<f64>) -> f64 {
    let err = (q5[0] - q4[0]).powi(2) + (q5[1] - q4[1]).powi(2);
    let tmp = epsilon*h_old/(2.0*err);
    let s = tmp.powf(1.0/4.0);
    s
}

/// Hmm what does this do? What acceleration does it calculate
fn acceleration(q: Vector4<f64>, param: &Potential) -> Vector2<f64> {
    let c_tmp = c(q, param);
    Vector2::new(c_tmp[2], c_tmp[3])
}

/// Performs a single iteration of the 'Verlet' integration method.
fn verlet(q: Vector4<f64>, h: f64, param: &Potential, a_old: Vector2<f64>) -> (Vector4<f64>, Vector4<f64>, Vector2<f64>) {
    let x_old = Vector2::new(q[0], q[1]);
    let v_old = Vector2::new(q[2], q[3]);
    let x_new = x_old + v_old*h + 0.5*a_old*h.powf(2.0);
    let a_new = acceleration(q, param);
    let v_new = v_old + 0.5*(a_old + a_new)*h;
    let q_new = Vector4::new(x_new[0], x_new[1], v_new[0], v_new[1]);
    (q_new, q_new, a_new)
}

/// Generates a random Gaussian surface. Arguments should be chosen carefully. 
/// The number of points must be odd.
/// 
/// This is the core algorithm that performs the generation fo the random surface
/// i.e. the parameters are those used by the central algorithm and do not make
/// much sense directly. Call this function only via the random_surf_gen method
/// of the RandSurface struct.
/// 
/// # Arguments
/// * `h` - Height parameter
/// * `dx` - Element length in the random surface
/// * `lambda` - Parameter related to the correlation length
/// * `s` - Standard deviation of the Gaussain (usually 1? can I get rid of it?)
/// * `n` - Number of elements, must be odd
fn random_surf_gen_core(h: f64, dx: f64, lambda: f64, s: f64, n: usize) ->  (Vec<f64>, Vec<f64>){
    assert!(n % 2 == 1);
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

    //let mut fs = convolve(zs, es);
    // Convert from vec to ndarray, it would be easier if this was all written in terms
    // of ndarray (but that might take a while, lets do that in steps) 
    let zs_a = Array1::from_shape_vec(n, zs).unwrap();
    let es_a = Array1::from_shape_vec(n, es).unwrap();
    let fs_a = fftconvolve(&zs_a, &es_a, Mode::Same).unwrap();
    let mut fs: Vec<f64> = vec![0.0; n];
    for i in 1..n {
        fs[i] = fs_a[i]*h;
    }

    (fs, xs)
}


//----------------------------------------------------------------------------//
//------------------------Random Surface struct-------------------------------//
//----------------------------------------------------------------------------//

pub struct RandSurface{
    lambda: f64,
    h: f64,
    h_rms: f64,
    corr_len: f64,
    dx: f64,
    pot: Potential
}

impl Default for RandSurface {
    fn default() -> Self {
        RandSurface { 
            lambda: 0.0,    // Correlation length variable used in the surface constructor
            h: 0.0,         // Height variable used in the surface constructor
            h_rms: 0.0,     // RMS height desired for the surface
            corr_len: 0.0,  // Correlation length desired for the surface
            dx: 0.0,        // Seperation between point in the x direction
            pot: Potential::default() // The potential
        }
    }
}

impl Clone for RandSurface {
    fn clone(&self) -> Self {
        RandSurface { 
            lambda: self.lambda, 
            h: self.h, 
            h_rms: self.h_rms, 
            corr_len: self.corr_len, 
            dx: self.dx,
            pot: self.pot.clone() 
        }
    }
}

impl std::fmt::Display for RandSurface {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
        // TODO
        write!(f, "I haven't done this")
    }
}

impl RandSurface {
    /// Generates a random Gaussian surface.
    /// 
    /// The surface has a Gaussian height distribution an exponential slope distribution
    /// and also has a Gaussian slope distribution. The surface generation is performed
    /// by the fun `random_surf_gen_Core()`. 
    /// 
    /// # Arguments
    /// * `h_rms` - the RMS height of the surface
    /// * `dx` - the length of elements in the surface
    /// * `corr_len` - the correlation length of the surface
    /// * `n` - the number of elements (must be odd)
    /// * `p` - potential paramters as an array [de, re, a]
    pub fn generate_surface(h_rms: f64, dx: f64, corr_len: f64, n: usize, p: [f64;3]) -> Self {
        let lambda: f64 = 0.5*corr_len.powf(2.0/3.0);
        let h: f64 = h_rms*((dx/lambda).tanh()).sqrt();
        let (fs, xs) = random_surf_gen_core(h, dx, lambda, 1.0, n);
        let mut surf: RandSurface = Default::default();
        surf.lambda = lambda;
        surf.h = h;
        surf.h_rms = h_rms;
        surf.corr_len = corr_len;
        surf.dx = dx;
        surf.pot = Potential::build_potential(&xs, &fs, p);
        surf
    }

    /// Saves a random surface to a text file.
    /// 
    /// The saved format is comma seperated with other parameters at the begnining of the
    /// file. The format can be ready by the python analysis scripts.
    pub fn save_to_file(&self, fname: &str) {
        let mut file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(fname)
            .unwrap();

        writeln!(file, "Surface statistics:").unwrap();
        writeln!(file, "h_RMS = {}", self.h_rms).unwrap();
        writeln!(file, "correlation_length = {}", self.corr_len).unwrap();
        writeln!(file, "Surface generation parameters").unwrap();
        writeln!(file, "h = {}", self.h).unwrap();
        writeln!(file, "Dx = {}", self.dx).unwrap();
        writeln!(file, "lambda = {}", self.lambda).unwrap();
        writeln!(file, "Surface points in space:").unwrap();
        writeln!(file, "x,y").unwrap();
        for i in 0..self.pot.xs.len() {
            if let Err(e) = writeln!(file, "{},{}", self.pot.xs[i], self.pot.ys[i]) {
                println!("Writing error: {}", e);
            }
        }
    }
}

//----------------------------------------------------------------------------//
//------------------------Potential struct------------------------------------//
//----------------------------------------------------------------------------//



/// Contains information on the potential field being used
///
/// * `de` - depth of the potential
/// * `re` - displacement of the centre of the potential well
/// * `a`  - width of the potential well
/// * `surf` - Spline representing the surface
/// * `gauss` - are we using the test GAussian bump sample
pub struct Potential {
    de: f64,
    re: f64,
    a: f64,
    surf: Spline<f64,f64>,
    xs: Vec<f64>,
    ys: Vec<f64>,
    gauss: bool
}

impl Clone for Potential {
    fn clone(&self) -> Self {
        Potential {
            de: self.de,
            re: self.re,
            a: self.a,
            surf: self.surf.clone(),
            xs: self.xs.clone(),
            ys: self.ys.clone(),
            gauss: self.gauss
        }
    }
}

impl std::fmt::Display for Potential {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
        write!(f, "\n======\n Potential\n======\nDe: {}\nre : {}\na : {}\nGauss bump? : {}",
            self.de, self.re, self.a, self.gauss)
    }
}

impl Potential {
    /// Constructs an interpolated potential struct using the points provided
    /// and the given parameters for the Morse function.
    fn build_potential(xs: &Vec<f64>, ys: &Vec<f64>, p: [f64;3]) -> Self {
        assert!(xs.len() == ys.len());
        let mut pot: Potential = Default::default();
        pot.surf = interpolate_surf(xs, ys);
        pot.xs = xs.clone();
        pot.ys = ys.clone();
        pot.de = p[0];
        pot.re = p[1];
        pot.a = p[2];
        pot.gauss = false;
        pot
    }

    /// Constructs a potential from a Gaussian bump for test purposes using the
    /// given parameters for the Morse function.
    pub fn gauss_potential(p: [f64;3]) -> Self {
        let mut pot: Potential = Default::default();
        pot.de = p[0];
        pot.re = p[1];
        pot.a = p[2];
        pot
    }
}

impl Default for Potential {
    fn default() -> Self {
        let ks = vec![Key::new(0.0, 0.0, Interpolation::Linear),
            Key::new(1.0, 0.0, Interpolation::Linear)];
        Potential {
            de: 0.5,
            re: 1.0,
            a: 1.0,
            surf: Spline::from_vec(ks),
            xs: vec![0.0],
            ys: vec![0.0],
            gauss: true
        }
    }
}



//----------------------------------------------------------------------------//
//---------------------------Atom struct--------------------------------------//
//----------------------------------------------------------------------------//



/// Representation of an atom in a potential with the ability to store the
/// history of the atoms trajectory.
pub struct Atom {
    /// The current state of the atom: its position and velocity
    q_current: Vector4<f64>,
    /// When using the R-K-F method the current error in the state of the atom
    error_current: Vector4<f64>,
    /// History of the state of the atom, recorded every `skip_record` steps
    q_history: Vec<Vector4<f64>>,
    /// History of the error in the method, recorded every `skip_record` steps
    error_history: Vec<Vector4<f64>>,
    /// Current iteration number
    current_it: usize,
    /// Are we recording the history of this atoms
    history: bool,
    /// Are we recording the history of the error of this atom
    record_error: bool,
    /// What fraction of iterations do we record, e.g. skip_record=3 records every 3rd
    skip_record: usize,
    /// What is the current acceleration of the atom
    current_acceleration: Vector2<f64>,
    /// What is the current time
    t_current: f64,
    /// What was the previous timestep
    h_previous: f64,
    /// History of the times of atom
    t_history: Vec<f64>
}

impl Default for Atom {
    fn default() -> Self {
        Atom {
            q_current: Vector4::new(f64::NAN, f64::NAN, f64::NAN, f64::NAN),
            error_current: Vector4::new(f64::NAN, f64::NAN, f64::NAN, f64::NAN),
            q_history: vec![Vector4::new(f64::NAN, f64::NAN, f64::NAN, f64::NAN); 0],
            error_history: vec![Vector4::new(f64::NAN, f64::NAN, f64::NAN, f64::NAN); 0],
            current_it: 0,
            history: false,
            skip_record: 10,
            record_error: false,
            current_acceleration: Vector2::new(0.0, 0.0),
            t_current: 0.0,
            h_previous: 0.01,
            t_history: vec![0.0]
        }
    }
}

impl Atom {
    /// Creates a new atom with the given starting conditions and vectors for
    /// recording its trajectory of the given length and sets up recording of
    /// the error if we are using R-K-F.
    fn new(q: Vector4<f64>, m: usize, record_error: bool, skip_record: usize,
            param: &Potential) -> Self {
        let mut at: Atom = Default::default();
        at.q_current = q;
        at.t_current = 0.0;
        at.skip_record = skip_record;
        if m != 0 {
            at.q_history = vec![at.q_current; m];
            at.t_history = vec![at.t_current; m];
            if record_error {
                at.record_error = true;
                at.error_history = vec![at.error_current; m];
            }
            at.history = true;
        }
        at.current_acceleration = acceleration(q, param);
        at
    }

    /// Adds a record of the atoms current state to the history of the atom
    ///
    /// # Arguments
    ///
    /// * `self` - the atom we are recording value for
    /// * `ind`  - which index should the state be saved into
    ///
    /// # Examples
    /// ```
    /// \\ Add the initial position to the first entry
    /// he_atom = Atom();
    /// he_atom.add_record(0);
    /// ```
    fn add_record(&mut self, ind: usize) {
        self.q_history[ind] = self.q_current;
        self.t_history[ind] = self.t_current;
        if self.record_error {
            self.error_history[ind] = self.error_current;
        }
    }

    /// Performs one iteration of the integration of the particle trajectory. 
    /// 
    /// Can use 3 different integration methods. Currently uses the same h for all 
    /// steps. TODO: a version of this that uses adaptive h.
    /// 
    /// # Arguments
    /// * `self` - the atom we are tracing the trajectory of
    /// * `h` - the timestep
    /// * `param` - information on the potential we are travelling in
    /// * `method` - which integration method to perform
    fn one_it(&mut self, h: f64, param: &Potential, method: &str) {
        let qs = match method {
            "Classic"  => runge_kutter_classic(self.q_current, h, param),
            "Fehlberg" => runge_kutter_fehlberg(self.q_current, h, param),
            "Verlet"   => verlet(self.q_current, h, param, self.current_acceleration),
            _          => panic!("Please provide an integration method that exists"),
        };
        self.q_current = qs.0;
        self.current_acceleration = qs.2;
        if self.record_error {
            self.error_current = qs.0 - qs.1;
        }
        self.current_it += 1;
        self.h_previous = h;
        if self.current_it % self.skip_record == 0 && self.history {
            let ind = self.current_it/self.skip_record;
            self.add_record(ind);
        }
    }

    /// Runs n iterations with the same timestep.
    /// 
    /// # Arguments
    /// * `h` - timestep
    /// * `n_it` - number of iterations to perform
    /// * `param` - information on the potential we are travelling in
    /// * `method` - which intergration method to perform
    /// * `height_stop` - a height above the surface to stop the integration
    fn run_n_iteration(&mut self, h: f64, n_it: usize, param: &Potential, method: &str, height_stop: f64) {
        for _i in 1..n_it {
            self.one_it(h, param, method);
            // If the atom is travelling upwards and passes the stop point then
            // stop simulating it
            if self.q_current[1] > height_stop && self.q_current[3] > 0.0 {
                break;
            }
        }
    }

    /// Runs iterations using adaptive runge-kutta
    fn adaptive_iterations(&mut self, h_max: f64, h_min: f64, n_max: usize, param: &Potential, height_stop: f64) {
        // TODO
    }

    /// Writes the recorded trajectory of an atom to a text file. 
    /// 
    /// Those timeseteps that have been recorded in the atom trajectory will be
    /// saved to a csv type file, The time, position, and velocity of the 
    /// particles are recorred. Assumes a constant timestep.
    /// 
    /// # Arguments
    /// * `f_name` - file name to save data into
    /// * `h` - timestep used in simulation
    pub fn write_trajectory(&self, f_name: &str, h: f64) {
        //println!("{}", f_name);
        let mut file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(f_name)
            .unwrap();

        if self.record_error {
            writeln!(file, "time,x,y,vx,vy,e_x,e_y,e_vx,e_vy").unwrap();
        } else {
            writeln!(file, "time,x,y,vx,vy").unwrap();
        }

        for i in 0..self.q_history.len() {
            let line = if self.record_error {
                format!("{},{},{},{},{},{},{},{},{}", self.t_history[i], self.q_history[i][0],
                    self.q_history[i][1], self.q_history[i][2], self.q_history[i][3],
                    self.error_history[i][0], self.error_history[i][1], self.error_history[i][2],
                    self.error_history[i][3])
            } else {
                format!("{},{},{},{},{}", self.t_history[i], self.q_history[i][0],
                    self.q_history[i][1], self.q_history[i][2], self.q_history[i][3])
            };
            writeln!(file, "{}", line).unwrap();
        }
    }
}

impl std::fmt::Display for Atom {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
        write!(f, "\n======\n Atom\n======\nQ : {}\nE : {}\nhistory : {}\nrecord_error : \
            {}\nrecord_size(q) : {}\nrecord_size(error) : {}",
            self.q_current, self.error_current, self.history, self.record_error,
            self.q_history.len(), self.error_history.len())
    }
}



//----------------------------------------------------------------------------//
//--------------------------SimParam struct-----------------------------------//
//----------------------------------------------------------------------------//



/// Contains parameters for a simulation for an atom. Combines them to tie them
/// together.
pub struct SimParam {
    pub h: f64,
    pub param: Potential,
    pub n_it: usize,
    pub method: String,
    pub skip_record: usize,
    pub height_stop: f64
}

impl std::fmt::Display for SimParam {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
        write!(f, "\n======\n SimParam\n======\nDt: {}\nn_it : {}\nmethod : {}\nPotential : {}\nHeight stop : {}",
            self.h, self.n_it, self.method, self.param, self.height_stop)
    }
}

//----------------------------------------------------------------------------//
//----------------------Primary simulation functions--------------------------//
//----------------------------------------------------------------------------//

/// Runs the trajectory of a single particle through the specified potential.
/// Returns an Atom struct containing the final state and potentially the
/// history of the trajectory.
///
/// # Argumnets
///
/// * `init_q` - Initial conditions, [x, y, vx, vy].
/// * `sim_params` - simulation paraneters.
/// * `record_atom` - Should the trajectory of the atom be recorded.
/// * `Potential` - Parameters for the potential to use.
pub fn run_particle(init_q: Vector4<f64>, sim_params: &SimParam, record_atom: bool) -> Atom {
    let m = if record_atom {((sim_params.n_it as f64)/(sim_params.skip_record as f64)).round() as usize} else {0};
    let record_error =  sim_params.method.eq(&String::from("Fehlberg"));
    if !record_error {
        assert!(sim_params.method.eq(&String::from("Classic")) ||
            sim_params.method.eq(&String::from("Verlet")));
    }
    let mut he = Atom::new(init_q, m, record_error, sim_params.skip_record, &sim_params.param);

    if he.history {
        he.add_record(0);
    }
    he.run_n_iteration(sim_params.h, sim_params.n_it, &sim_params.param, &sim_params.method,
                       sim_params.height_stop);
    he
}

/// Saves the provided position and velocity vectors to the provided file name
/// as comma delimated text files.
pub fn save_pos_vel(q: &[Vector4<f64>], f_name: &str) {
    let mut file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(f_name)
            .unwrap();
    writeln!(file, "x,y,v_x,v_y").unwrap();


    for at in q {
        let line = format!("{},{},{},{}", at[0], at[1], at[2], at[3]);
        writeln!(file, "{}", line).unwrap();
    }
}

/// Composes names for text files for the ith atom
pub fn compose_save_name(fname: &str, i: usize) -> String {
    let ith: String = i.to_string();
    let mut num: String = String::from("0").repeat(8 - ith.len());//.push_str(&ith);
    num.push_str(&ith);
    let mut save_name = fname.to_string();
    save_name.push_str("/atom");
    save_name.push_str(&num);
    save_name.push_str(".csv");
    save_name
}

/// Runs a simulation for a single atom
pub fn run_it(q_init: Vector4<f64>, i: usize, record: usize, fname: &str, sim_params: &SimParam) -> Vector4<f64> {
    let record_atom = i % record == 0;
    let he_atom = run_particle(q_init, sim_params, record_atom);
    // Only save some of the trajectories atoms
    if record_atom {
        let save_name = compose_save_name(fname, i);
        he_atom.write_trajectory(&save_name, sim_params.h);
    }

    he_atom.q_current
}

/// Runs the simulation on n atoms with the given list of starting conditions, parallelised
pub fn run_n_particle(record: usize, sim_params: &SimParam, fname: &str, init_q: Vec<Vector4<f64>>) -> Vec<Vector4<f64>> {
    let q_iter = init_q.into_par_iter().enumerate().map(|(i, x)| run_it(x, i, record, fname, sim_params));
    q_iter.collect()
}

/// Takes 4 slices and composes them into a Vec of 4-Vectors
pub fn compose_vector4(x: &[f64], y: &[f64], vx: &[f64], vy: &[f64]) -> Vec<Vector4<f64>> {
    assert!(x.len() == y.len() && y.len() == vx.len() && vx.len() == vy.len());
    let n = x.len();
    let mut qs = vec![Vector4::new(0.0, 0.0, 0.0, 0.0); n];
    // TODO: can be done as an iterator?
    for i in 0..n {
        qs[i] = Vector4::new(x[i], y[i], vx[i], vy[i]);
    }
    qs
}



//----------------------------------------------------------------------------//
//------------------------External functions----------------------------------//
//----------------------------------------------------------------------------//
// These functions create a shared library that can be called from other
// languages, e.g. python.
// I intend to stop using these

#[no_mangle]
pub unsafe extern fn single_particle(x: f64, y: f64, vx: f64, vy: f64, h: f64,
        n_it: u64, text: *const u8, len: u64, p1: *const f64, method: *const u8,
        len2: u64, x_surf: *const f64, y_surf: *const f64, n_surf: u64,
        test_surf: u64, skip_record: u64, height_stop: f64) {
    let init_q = Vector4::new(x, y, vx, vy);

    // Have to go via C-string to get the file name
    assert!(!text.is_null());
    let c_str = { slice::from_raw_parts(text, len as usize) };
    let fname = str::from_utf8(&c_str).unwrap();

    // Have to go via C-string to get the integration method
    assert!(!method.is_null());
    let c_str2 = { slice::from_raw_parts(method, len2 as usize) };
    let method = str::from_utf8(&c_str2).unwrap();

    // The parameters of the potential
    assert!(!p1.is_null());
    let p2 = { slice::from_raw_parts(p1, 3) };
    let param = if test_surf == 1 {
        Potential::gauss_potential(copy_into_array(p2))
    } else {
        assert!(!x_surf.is_null());
        let x_s = { slice::from_raw_parts(x_surf, n_surf as usize) };
        assert!(!y_surf.is_null());
        let y_s = { slice::from_raw_parts(y_surf, n_surf as usize) };
        Potential::build_potential(&x_s.to_vec(), &y_s.to_vec(), copy_into_array(p2))
    };

    let sim_params = SimParam {
        skip_record: skip_record as usize,
        h,
        param,
        n_it: n_it as usize,
        method: method.to_string(),
        height_stop
    };

    let he_atom = run_particle(init_q, &sim_params, true);
    let save_fname: String = fname.to_string();
    he_atom.write_trajectory(&save_fname, h);
}

#[no_mangle]
pub unsafe extern fn multiple_particle(xs: *const f64, ys: *const f64, vxs: *const f64,
        vys: *const f64, n_atom: u64, h: f64, n_it: u64, text: *const u8, len: u64,
        p1: *const f64, record: u64, method: *const u8, len2: u64, x_surf: *const f64,
        y_surf: *const f64, n_surf: u64, test_surf: u64, skip_record: u64, height_stop: f64) {
    // Get a proper rust array from what we are passed
    let n = n_atom as usize;
    assert!(!xs.is_null());
    let x = { slice::from_raw_parts(xs, n) };
    assert!(!ys.is_null());
    let y = { slice::from_raw_parts(ys, n) };
    assert!(!vxs.is_null());
    let vx = { slice::from_raw_parts(vxs, n) };
    assert!(!vys.is_null());
    let vy = { slice::from_raw_parts(vys, n) };

    // The parameters of the potential
    assert!(!p1.is_null());
    let p2 = { slice::from_raw_parts(p1, 3) };
    let param = if test_surf == 1 {
        Potential::gauss_potential(copy_into_array(p2))
    } else {
        assert!(!x_surf.is_null());
        let x_s = { slice::from_raw_parts(x_surf, n_surf as usize) };
        assert!(!y_surf.is_null());
        let y_s = { slice::from_raw_parts(y_surf, n_surf as usize) };
        Potential::build_potential(&x_s.to_vec(), &y_s.to_vec(), copy_into_array(p2))
    };

    // Have to go via C-string to get the file name
    assert!(!text.is_null());
    let c_str = { slice::from_raw_parts(text, len as usize) };
    let fname = str::from_utf8(&c_str).unwrap();

    // Have to go via C-string to get the integration method
    assert!(!method.is_null());
    let c_str2 = { slice::from_raw_parts(method, len2 as usize) };
    let method = str::from_utf8(&c_str2).unwrap();

    let sim_params = SimParam {
        skip_record: skip_record as usize,
        h,
        param,
        n_it: n_it as usize,
        method: method.to_string(),
        height_stop
    };

    let init_q = compose_vector4(x, y, vx, vy);
    let final_q = run_n_particle(record as usize, &sim_params, &fname.to_string(), init_q);

    // Save all final positions and directions
    let mut final_fname: String = fname.to_string();
    final_fname.push_str("/final_states.csv");
    save_pos_vel(&final_q, &final_fname);
}

#[no_mangle]
pub unsafe extern fn calc_potential(xs: *const f64, ys: *const f64, vs: *mut f64,
        len: u64, p1: *const f64, x_surf: *const f64, y_surf: *const f64,
        n_surf: u64, test_surf: u64) {
    // Get a proper rust array from what we are passed
    let l = len as usize;
    assert!(!xs.is_null());
    let x = { slice::from_raw_parts(xs, l) };
    assert!(!ys.is_null());
    let y = { slice::from_raw_parts(ys, l) };
    assert!(!ys.is_null());
    let v = { slice::from_raw_parts_mut(vs, l) };

    // The parameters of the potential
    assert!(!p1.is_null());
    let p2 = { slice::from_raw_parts(p1, 3) };
    let param = if test_surf == 1 {
        Potential::gauss_potential(copy_into_array(p2))
    } else {
        assert!(!x_surf.is_null());
        let x_s = { slice::from_raw_parts(x_surf, n_surf as usize) };
        assert!(!y_surf.is_null());
        let y_s = { slice::from_raw_parts(y_surf, n_surf as usize) };
        Potential::build_potential(&x_s.to_vec(), &y_s.to_vec(), copy_into_array(p2))
    };

    // TODO: can be done as an iterator
    for i in 0..l {
        v[i] = if param.gauss {
            morse_gauss(x[i], y[i], &param)
        } else {
            morse(x[i], y[i], &param)
        };
    }
}

#[no_mangle]
pub unsafe extern fn gauss_bump(xs: *const f64, ys: *mut f64, len: u64, sigma: f64) {
    // Get a proper rust array from what we are passed
    let l = len as usize;
    assert!(!xs.is_null());
    let x = { slice::from_raw_parts(xs, l) };
    assert!(!ys.is_null());
    let y = { slice::from_raw_parts_mut(ys, l) };

    let h: f64 = 30.0;
    // TODO: can be done as an iterator?
    for i in 1..l {
        y[i] = gaussian(x[i], sigma, h);
    }
}


#[no_mangle]
pub unsafe extern fn print_text(text: *const u8, len: u64) {
    assert!(!text.is_null());
    let c_str = { slice::from_raw_parts(text, len as usize) };
    let string = str::from_utf8(&c_str).unwrap();
    println!("{}", len);
    println!("{}", string);
}

#[no_mangle]
/// The arrays of f64 `xs` and `ys` must both be of length `len`.
/// The arrays of f64 `new_x` and `new_y` must both be of length `n_interp`.
/// The array `new_y` will get overwritten.
pub unsafe extern fn interpolate_test(xs: *const f64, ys: *mut f64, len: u64,
        new_x: *const f64, new_y: *mut f64, n_interp: u64) {
    let l = len as usize;
    assert!(!xs.is_null());
    let x = { slice::from_raw_parts(xs, l) };
    assert!(!ys.is_null());
    let y = { slice::from_raw_parts(ys, l) };

    let m = n_interp as usize;
    assert!(!new_x.is_null());
    let xi = { slice::from_raw_parts(new_x, m) };
    assert!(!new_y.is_null());
    let yi = { slice::from_raw_parts_mut(new_y, m) };

    let xx = x.to_vec();
    let yy = y.to_vec();

    let splin = interpolate_surf(&xx, &yy);
    println!("{:?}", splin);

    // TODO: can be done as an iterator?
    for i in 0..m {
        yi[i] = match splin.clamped_sample(xi[i]) {
            Some(val) => val,
            None     => f64::NAN
        };
        println!("({}, {})", xi[i], yi[i]);
    }
}
