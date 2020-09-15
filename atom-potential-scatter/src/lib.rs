extern crate nalgebra as na;

use na::{Vector4, Matrix4, Vector2};
use core::f64::consts::PI;
use std::slice;
use std::fs::OpenOptions;
use std::io::prelude::*;
use std::str;
use splines::{Interpolation, Key, Spline};

// Constans used by the R-K-F integration method.
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

fn gaussian(x: f64, s: f64, height: f64) -> f64{
    height*(1.0/(s*(2.0*PI).sqrt()))*(-x.powi(2)/(2.0*s.powi(2))).exp()
}

fn morse(x: f64, y: f64, p: &Potential) -> f64 {
    let r = y - gaussian(x, 10.0, 60.0);
    p.de*( (-2.0*p.a*(r - p.re)).exp() - 2.0*(-p.a*(r - p.re)).exp() )
}

fn diff_at_point(x: f64) -> f64 {
    let epsilon: f64 = 0.01;
    (gaussian(x + 0.5*epsilon, 10.0, 60.0) - gaussian(x - 0.5*epsilon, 10.0, 60.0))/epsilon
}

fn c(q: Vector4<f64>, p: &Potential) -> Vector4<f64> {
    let x = q[0];
    let y = q[1];
    let r = y - gaussian(x, 10.0, 60.0);
    let dd = diff_at_point(x);
    let ax = 2.0*p.a*p.de*dd*( (-2.0*p.a*(r - p.re)).exp() - (-p.a*(r - p.re)).exp() );
    let ay = p.de*( -2.0*p.a*(-2.0*p.a*(r - p.re)).exp() + 2.0*p.a*(-p.a*(r - p.re)).exp() );
    Vector4::new(0.0, 0.0, -ax, -ay)
}

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

struct Potential {
    de: f64,
    re: f64,
    a: f64
}

struct Atom {
    q_current: Vector4<f64>,
    error_current: Vector4<f64>,
    mass: f64,
    x_history: Vec<f64>,
    y_history: Vec<f64>,
    vx_history: Vec<f64>,
    vy_history: Vec<f64>,
    error_history: Vec<Vector4<f64>>,
    current_it: usize,
    history: bool,
    record_error: bool
}

impl Atom {
    fn add_record(&mut self, ind: usize) {
        self.x_history[ind] = self.q_current[0];
        self.y_history[ind] = self.q_current[1];
        self.vx_history[ind] = self.q_current[2];
        self.vy_history[ind] = self.q_current[3];
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
        self.current_it = self.current_it + 1;
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
    
    fn write_trajectory(&self, f_name: &String, h: f64) {
        let mut file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(f_name)
            .unwrap();
        
        if self.record_error {
            writeln!(file, "{}", "time,x,y,vx,vy,e_x,e_y,e_vx,e_vy").unwrap();
        } else {
            writeln!(file, "{}", "time,x,y,vx,vy").unwrap();
        }
        
        let mut t: f64 = 0.0;
        for i in 0..self.x_history.len() {
            let line = if self.record_error {
                format!("{},{},{},{},{},{},{},{},{}", t, self.x_history[i],
                    self.y_history[i], self.vx_history[i], self.vy_history[i], 
                    self.error_history[i][0], self.error_history[i][1], self.error_history[i][2],
                    self.error_history[i][3])
            } else {
                format!("{},{},{},{},{}", t, self.x_history[i],
                    self.y_history[i], self.vx_history[i], self.vy_history[i])
            };
            writeln!(file, "{}", line).unwrap();
            t += h;
        }
    }
}

impl std::fmt::Display for Atom {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
        write!(f, "\n======\n Atom\n======\nQ : {}\nE : {}\nhistory : {}\nrecord_error : \
            {}\nrecord_size(x) : {}\nrecord_size(y) : {}\nrecord_size(vx) : {} \
            \nrecord_size(vy) : {}\nrecord_size(error) : {}", 
            self.q_current, self.error_current, self.history, self.record_error, 
            self.x_history.len(), self.y_history.len(), self.vx_history.len(), 
            self.vy_history.len(), self.error_history.len())
    }
}

fn run_particle(init_x: Vector2<f64>, init_v: Vector2<f64>, h: f64, n: usize, 
        param: &Potential, record_atom: bool, method: &String) -> Atom {
    let m = if record_atom {((n as f64)/10.0).round() as usize} else {0};
    let record_error =  method.eq(&String::from("Fehlberg"));
    if !record_error {
        assert!(method.eq(&String::from("Classic")));
    }
    let m_e = if record_error {m} else {0};
    let mut he = Atom {
        q_current: Vector4::new(init_x[0], init_x[1], init_v[0], init_v[1]),
        error_current: Vector4::new(0.0, 0.0, 0.0, 0.0),
        mass: 1.0,
        x_history: vec![0.0; m],
        y_history: vec![0.0; m],
        vx_history: vec![0.0; m],
        vy_history: vec![0.0; m],
        error_history: vec![Vector4::new(0.0, 0.0, 0.0, 0.0); m_e],
        current_it: 0,
        history: record_atom,
        record_error: record_error
    };
    
    if he.history {
        he.add_record(0);
    }
    he.run_n_iteration(h, n, param);
    he
}

fn save_pos_vel(pos: &[Vector2<f64>], vel: &[Vector2<f64>], f_name: &String) {
    let mut file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(f_name)
            .unwrap();
    writeln!(file, "{}", "atom,,x,y,v_x,v_y").unwrap();
    
    for i in 0..pos.len() {
        let line = format!("{},{},{},{},{}", i, pos[i][0], pos[i][1], vel[i][0], 
                           vel[i][1]);
        writeln!(file, "{}", line).unwrap();
    }
}

fn compose_save_name(fname: &String, i: usize) -> String {
    let ith: String = i.to_string();
    let mut num: String = String::from("0").repeat(8 - ith.len());//.push_str(&ith);
    num.push_str(&ith);
    let mut save_name = fname.to_string();
    save_name.push_str("/atom");
    save_name.push_str(&num);
    save_name.push_str(".csv");
    save_name
}

//fn interpolate_surf(xs: &[f64], ys: &[f64]) -> Spline<f64, f64> {
//
//}

//-----------------------------------------------------------------------------

#[no_mangle]
pub extern fn single_particle(x: f64, y: f64, vx: f64, vy: f64, h: f64, 
        n_it: u64, text: *const u8, len: u64, p1: *const f64, method: *const u8, len2: u64) {
    let init_x = Vector2::new(x, y);
    let init_v = Vector2::new(vx, vy);
    
    // Have to go via C-string to get the file name
    assert!(!text.is_null());
    let c_str = unsafe { slice::from_raw_parts(text, len as usize) };
    let fname = str::from_utf8(&c_str).unwrap();
    
    // Have to go via C-string to get the integration method
    assert!(!method.is_null());
    let c_str2 = unsafe { slice::from_raw_parts(method, len2 as usize) };
    let method = str::from_utf8(&c_str2).unwrap();
    
    // The parameters of the potential
    assert!(!p1.is_null());
    let p2 = unsafe { slice::from_raw_parts(p1, 3) };
    let param = Potential{
        de: p2[0],
        re: p2[1],
        a: p2[2]
    };
    
    let he_atom = run_particle(init_x, init_v, h, n_it as usize, &param, true, &method.to_string());
    let mut save_fname: String = fname.to_string();
    save_fname.push_str("/atom_trajectory.csv");
    he_atom.write_trajectory(&fname.to_string(), h);
}

#[no_mangle]
pub extern fn multiple_particle(xs: *const f64, ys: *const f64, vxs: *const f64, 
        vys: *const f64, n_atom: u64, h: f64, n_it: usize, text: *const u8, len: u64, 
        p1: *const f64, record: u64, method: *const u8, len2: u64) {
    // Get a proper rust array from what we are passed
    let n = n_atom as usize;
    assert!(!xs.is_null());
    let x = unsafe { slice::from_raw_parts(xs, n) };
    assert!(!ys.is_null());
    let y = unsafe { slice::from_raw_parts(ys, n) };
    assert!(!vxs.is_null());
    let vx = unsafe { slice::from_raw_parts(vxs, n) };
    assert!(!vys.is_null());
    let vy = unsafe { slice::from_raw_parts(vys, n) };
    
    // The parameters of the potential
    assert!(!p1.is_null());
    let p2 = unsafe { slice::from_raw_parts(p1, 3) };
    let param = Potential{
        de: p2[0],
        re: p2[1],
        a: p2[2]
    };
    
    // Have to go via C-string to get the file name
    assert!(!text.is_null());
    let c_str = unsafe { slice::from_raw_parts(text, len as usize) };
    let fname = str::from_utf8(&c_str).unwrap();
    
    // Have to go via C-string to get the integration method
    assert!(!method.is_null());
    let c_str2 = unsafe { slice::from_raw_parts(method, len2 as usize) };
    let method = str::from_utf8(&c_str2).unwrap();
    
    // Data structures for all final positions and direction
    let mut final_pos =  vec![Vector2::new(0.0,0.0); n_atom as usize];
    let mut final_v = vec![Vector2::new(0.0,0.0); n_atom as usize];
    
    for i in 1..n {
        let record_atom = i % record as usize == 0;
        let init_x = Vector2::new(x[i], y[i]);
        let init_v = Vector2::new(vx[i], vy[i]);
        
        let he_atom = run_particle(init_x, init_v, h, n_it, &param, record_atom, &method.to_string());
        // Only save some of the trajectories atoms
        if record_atom {
            let save_name = compose_save_name(&fname.to_string(), i);
            he_atom.write_trajectory(&save_name, h);
        }
        
        final_pos[i][0] = he_atom.q_current[0];
        final_pos[i][1] = he_atom.q_current[1];
        final_v[i][0] = he_atom.q_current[2];
        final_v[i][1] = he_atom.q_current[3];
    }
    
    // Save all final positions and directions
    let mut final_fname: String = fname.to_string();
    final_fname.push_str("/final_states.csv");
    save_pos_vel(&final_pos, &final_v, &final_fname);
}

#[no_mangle]
pub extern fn calc_potential(xs: *const f64, ys: *const f64, vs: *mut f64, len: u64, p1: *const f64) {
    // Get a proper rust array from what we are passed
    let l = len as usize;
    assert!(!xs.is_null());
    let x = unsafe { slice::from_raw_parts(xs, l) };
    assert!(!ys.is_null());
    let y = unsafe { slice::from_raw_parts(ys, l) };
    assert!(!ys.is_null());
    let v = unsafe { slice::from_raw_parts_mut(vs, l) };
    
    // The parameters of the potential
    assert!(!p1.is_null());
    let p2 = unsafe { slice::from_raw_parts(p1, 3) };
    let param = Potential{
        de: p2[0],
        re: p2[1],
        a: p2[2]
    };
    
    for i in 1..l {
        v[i] = morse(x[i], y[i], &param);
    }
}

#[no_mangle]
pub extern fn gauss_bump(xs: *const f64, ys: *mut f64, len: u64, s: f64) {
    // Get a proper rust array from what we are passed
    let l = len as usize;
    assert!(!xs.is_null());
    let x = unsafe { slice::from_raw_parts(xs, l) };
    assert!(!ys.is_null());
    let y = unsafe { slice::from_raw_parts_mut(ys, l) };
    
    let h: f64 = 30.0;
    for i in 1..l {
        y[i] = gaussian(x[i], s, h);
    }
}


#[no_mangle]
pub extern fn print_text(text: *const u8, len: u64) {
    assert!(!text.is_null());
    let c_str = unsafe { slice::from_raw_parts(text, len as usize) };
    let string = str::from_utf8(&c_str).unwrap();
    println!("{}", len);
    println!("{}", string);
}
