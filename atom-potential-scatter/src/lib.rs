#![crate_name = "atom_potential_scatter"]

extern crate nalgebra as na;
extern crate rayon;

use na::{Vector4, Matrix4};
use core::f64::consts::PI;
use std::slice;
use std::fs::OpenOptions;
use std::io::prelude::*;
use std::str;
use std::convert::AsMut;
use splines::{Interpolation, Key, Spline};
use rayon::prelude::*;

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
fn interpolate_surf(xs: Vec<f64>, ys: Vec<f64>) -> Spline<f64,f64> {
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
    let r = y - gaussian(x, 10.0, 60.0);
    p.de*( (-2.0*p.a*(r - p.re)).exp() - 2.0*(-p.a*(r - p.re)).exp() )
}

fn diff_at_point_gauss(x: f64) -> f64 {
    let epsilon: f64 = 0.01;
    (gaussian(x + 0.5*epsilon, 10.0, 60.0) - gaussian(x - 0.5*epsilon, 10.0, 60.0))/epsilon
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

fn c(q: Vector4<f64>, p: &Potential) -> Vector4<f64> {
    let x = q[0];
    let y = q[1];
    let r = if p.gauss {
        y - gaussian(x, 10.0, 60.0)
    } else {
        y - match p.surf.clamped_sample(x) {
            Some(val) => val,
            None      => f64::NAN
        }
    };
    let dd = if p.gauss {
        diff_at_point_gauss(x)
    } else {
        diff_at_point(x, &p.surf)
    };
    let ax = 2.0*p.a*p.de*dd*( (-2.0*p.a*(r - p.re)).exp() - (-p.a*(r - p.re)).exp() );
    let ay = p.de*( -2.0*p.a*(-2.0*p.a*(r - p.re)).exp() + 2.0*p.a*(-p.a*(r - p.re)).exp() );
    Vector4::new(0.0, 0.0, -ax, -ay)
}

/// Performs a single iteration of the 'classic' Runge-Kutta method
fn runge_kutter_classic(q: Vector4<f64>, h: f64, param: &Potential) -> Vector4<f64> {
    let b = Matrix4::new(0.0, 0.0, 1.0, 0.0, 
                         0.0, 0.0, 0.0, 1.0, 
                         0.0, 0.0, 0.0, 0.0, 
                         0.0, 0.0, 0.0, 0.0);
    let k1 = b*q + c(q, param);
    let k2 = b*q + c(q + h*k1/2.0, param);
    let k3 = b*q + c(q + h*k2/2.0, param);
    let k4 = b*q + c(q + h*k3, param);
    q + (1.0/6.0)*h*(k1 + 2.0*k2 + 2.0*k3 + k4)
}

/// Performs a single iteration of the 'Runge-Kutta-Fehlberg' method. Returns
/// a tuple containing the results from both the 5th and 4th order methods
fn runge_kutter_fehlberg(q: Vector4<f64>, h: f64, param: &Potential) -> (Vector4<f64>, Vector4<f64>) {
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
    (q_5th, q_4th)
}



//----------------------------------------------------------------------------//
//----------------------------------------------------------------------------//
//----------------------------------------------------------------------------//



/// Contains information on the potential field being used
struct Potential {
    de: f64,
    re: f64,
    a: f64,
    surf: Spline<f64,f64>,
    gauss: bool
}

impl Clone for Potential {
    fn clone(&self) -> Self {
        Potential {
            de: self.de,
            re: self.re,
            a: self.a,
            surf: self.surf.clone(),
            gauss: self.gauss
        }
    }
}

impl Potential {
    /// Constructs an interpolated potential struct using the points provided
    /// and the given parameters for the Morse function.
    fn build_potential(xs: Vec<f64>, ys: Vec<f64>, p: [f64;3]) -> Self {
        assert!(xs.len() == ys.len());
        let mut pot: Potential = Default::default();
        pot.surf = interpolate_surf(xs, ys);
        pot.de = p[0];
        pot.re = p[1];
        pot.a = p[2];
        pot.gauss = false;
        pot
    }
    
    /// Constructs a potential from a Gaussian bump for test purposes using the
    /// given parameters for the Morse function.
    fn gauss_potential(p: [f64;3]) -> Self {
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
            gauss: true
        }
    }
}



//----------------------------------------------------------------------------//
//----------------------------------------------------------------------------//
//----------------------------------------------------------------------------//



/// Representation of an atom in a potential with the ability to store the
/// history of the atoms trajectory.
struct Atom {
    /// The current state of the atom: its position and velocity
    q_current: Vector4<f64>,
    /// When using the R-K-F method the current error in the state of the atom
    error_current: Vector4<f64>,
    q_history: Vec<Vector4<f64>>,
    error_history: Vec<Vector4<f64>>,
    current_it: usize,
    history: bool,
    record_error: bool
}

impl Default for Atom {
    fn default() -> Self {
        Atom {
            q_current: Vector4::new(0.0, 0.0, 0.0, 0.0),
            error_current: Vector4::new(0.0, 0.0, 0.0, 0.0),
            q_history: vec![Vector4::new(0.0, 0.0, 0.0, 0.0); 0],
            error_history: vec![Vector4::new(0.0, 0.0, 0.0, 0.0); 0],
            current_it: 0,
            history: false,
            record_error: false
        }
    }
}

impl Atom {
    /// Creates a new atom with the given starting conditions and vectors for
    /// recording its trajectory of the given length and sets up recording of
    /// the error if we are using R-K-F.
    fn new(q: Vector4<f64>, m: usize, record_error: bool) -> Self {
        let mut at: Atom = Default::default();
        at.q_current = q;
        if m != 0 {
            at.q_history = vec![at.error_current; m];
            if record_error {
                at.record_error = true;
                at.error_history = vec![at.error_current; m];
            }
            at.history = true;
        }
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
        if self.record_error {
            self.error_history[ind] = self.error_current;
        }
    }
    
    fn one_it(&mut self, h: f64, param: &Potential) {
        let qs = if self.record_error {
            runge_kutter_fehlberg(self.q_current, h, param)
        } else {
            (runge_kutter_classic(self.q_current, h, param), Vector4::new(0.0, 0.0, 0.0, 0.0))
        };
        self.q_current = qs.0;
        self.error_current = qs.0 - qs.1;
        self.current_it += 1;
        if self.current_it % 10 == 0 && self.history {
            let ind = self.current_it/10;
            self.add_record(ind);
        }
    }
    
    fn run_n_iteration(&mut self, h: f64, n_it: usize, param: &Potential) {
        for _i in 1..n_it {
            self.one_it(h, param);
        }
    }
    
    fn write_trajectory(&self, f_name: &str, h: f64) {
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
        
        let mut t: f64 = 0.0;
        for i in 0..self.q_history.len() {
            let line = if self.record_error {
                format!("{},{},{},{},{},{},{},{},{}", t, self.q_history[i][0],
                    self.q_history[i][1], self.q_history[i][2], self.q_history[i][3], 
                    self.error_history[i][0], self.error_history[i][1], self.error_history[i][2],
                    self.error_history[i][3])
            } else {
                format!("{},{},{},{},{}", t, self.q_history[i][0],
                    self.q_history[i][1], self.q_history[i][2], self.q_history[i][3])
            };
            writeln!(file, "{}", line).unwrap();
            t += h;
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
//----------------------------------------------------------------------------//
//----------------------------------------------------------------------------//



/// Contains parameters for a simulation for an atom. Combines them to tie them
/// together.
struct SimParam {
    h: f64,
    param: Potential,
    n_it: usize,
    method: String
}

/// Runs the trajectory of a single particle through the specified potential. 
/// Returns an Atom struct containing the final state and potentially the
/// history of the trajectory.
/// 
/// # Argumnets
/// 
/// * `init_x` - Initial position.
/// * `init_v` - Initial velocity.
/// * `h` - Timestep to use.
/// * `n` - The number of time steps to run.
/// * `Potential` - Parameters for the potential to use.
/// * `record_atom` - Flag for if to record the history of the atom
/// * `method` - The integration method to use, either "Fehlberg" or "Classic"
fn run_particle(init_q: Vector4<f64>, sim_params: &SimParam, record_atom: bool) -> Atom {
    let m = if record_atom {((sim_params.n_it as f64)/10.0).round() as usize} else {0};
    let record_error =  sim_params.method.eq(&String::from("Fehlberg"));
    if !record_error {
        assert!(sim_params.method.eq(&String::from("Classic")));
    }
    let mut he = Atom::new(init_q, m, record_error);
    
    if he.history {
        he.add_record(0);
    }
    he.run_n_iteration(sim_params.h, sim_params.n_it, &sim_params.param);
    he
}

/// Saves the provided position and velocity vectors to the provided file name
/// as comma delimated text files.
fn save_pos_vel(q: &[Vector4<f64>], f_name: &str) {
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
fn compose_save_name(fname: &str, i: usize) -> String {
    let ith: String = i.to_string();
    let mut num: String = String::from("0").repeat(8 - ith.len());//.push_str(&ith);
    num.push_str(&ith);
    let mut save_name = fname.to_string();
    save_name.push_str("/atom");
    save_name.push_str(&num);
    save_name.push_str(".csv");
    save_name
}

fn run_it(q_init: Vector4<f64>, i: usize, record: usize, fname: &str, sim_params: &SimParam) -> Vector4<f64> {
    let record_atom = i % record == 0;
    let he_atom = run_particle(q_init, sim_params, record_atom);
    // Only save some of the trajectories atoms
    if record_atom {
        let save_name = compose_save_name(fname, i);
        he_atom.write_trajectory(&save_name, sim_params.h);
    }
    
    he_atom.q_current
}

/// Runs the simulation on n atoms with the given list of starting conditions
fn run_n_particle(record: usize, sim_params: &SimParam, fname: &str, init_q: Vec<Vector4<f64>>) -> Vec<Vector4<f64>> {
    let q_iter = init_q.into_par_iter().enumerate().map(|(i, x)| run_it(x, i, record, fname, sim_params));
    q_iter.collect()
}

/// Takes 4 slices and composes them into a Vec of 4-Vectors
fn compose_vector4(x: &[f64], y: &[f64], vx: &[f64], vy: &[f64]) -> Vec<Vector4<f64>> {
    assert!(x.len() == y.len() && y.len() == vx.len() && vx.len() == vy.len());
    let n = x.len();
    let mut qs = vec![Vector4::new(0.0, 0.0, 0.0, 0.0); n];
    for i in 0..n {
        qs[i] = Vector4::new(x[i], y[i], vx[i], vy[i]);
    }
    qs
}



//----------------------------------------------------------------------------//
//----------------------------------------------------------------------------//
//----------------------------------------------------------------------------//



#[no_mangle]
#[doc(saftey)]
pub unsafe extern fn single_particle(x: f64, y: f64, vx: f64, vy: f64, h: f64, 
        n_it: u64, text: *const u8, len: u64, p1: *const f64, method: *const u8, 
        len2: u64) {
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
    let param = Potential::gauss_potential(copy_into_array(p2));
    
    let sim_params = SimParam {
        h,
        param,
        n_it: n_it as usize,
        method: method.to_string()
    };
    
    let he_atom = run_particle(init_q, &sim_params, true);
    let mut save_fname: String = fname.to_string();
    save_fname.push_str("/atom_trajectory.csv");
    he_atom.write_trajectory(&fname.to_string(), h);
}

#[no_mangle]
#[doc(saftey)]
pub unsafe extern fn multiple_particle(xs: *const f64, ys: *const f64, vxs: *const f64, 
        vys: *const f64, n_atom: u64, h: f64, n_it: u64, text: *const u8, len: u64, 
        p1: *const f64, record: u64, method: *const u8, len2: u64, x_surf: *const f64, 
        y_surf: *const f64, n_surf: u64, test_surf: u64) {
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
        Potential::build_potential(x_s.to_vec(), y_s.to_vec(), copy_into_array(p2))
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
        h,
        param,
        n_it: n_it as usize,
        method: method.to_string()
    };
    
    let init_q = compose_vector4(x, y, vx, vy);
    let final_q = run_n_particle(record as usize, &sim_params, &fname.to_string(), init_q);
    
    // Save all final positions and directions
    let mut final_fname: String = fname.to_string();
    final_fname.push_str("/final_states.csv");
    save_pos_vel(&final_q, &final_fname);
}

#[no_mangle]
#[doc(saftey)]
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
        Potential::build_potential(x_s.to_vec(), y_s.to_vec(), copy_into_array(p2))
    };
    
    for i in 1..l {
        v[i] = if param.gauss {
            morse_gauss(x[i], y[i], &param)
        } else {
            morse(x[i], y[i], &param)
        };
    }
}

#[no_mangle]
#[doc(saftey)]
pub unsafe extern fn gauss_bump(xs: *const f64, ys: *mut f64, len: u64, sigma: f64) {
    // Get a proper rust array from what we are passed
    let l = len as usize;
    assert!(!xs.is_null());
    let x = { slice::from_raw_parts(xs, l) };
    assert!(!ys.is_null());
    let y = { slice::from_raw_parts_mut(ys, l) };
    
    let h: f64 = 30.0;
    for i in 1..l {
        y[i] = gaussian(x[i], sigma, h);
    }
}


#[no_mangle]
#[doc(saftey)]
pub unsafe extern fn print_text(text: *const u8, len: u64) {
    assert!(!text.is_null());
    let c_str = { slice::from_raw_parts(text, len as usize) };
    let string = str::from_utf8(&c_str).unwrap();
    println!("{}", len);
    println!("{}", string);
}

#[no_mangle]
#[doc(saftey)]
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
    
    let splin = interpolate_surf(xx, yy);
    println!("{:?}", splin);
    
    for i in 0..m {
        yi[i] = match splin.clamped_sample(xi[i]) {
            Some(val) => val,
            None     => f64::NAN
        };
        println!("({}, {})", xi[i], yi[i]);
    }
}
