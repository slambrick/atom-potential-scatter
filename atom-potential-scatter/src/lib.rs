extern crate nalgebra as na;

use na::{Vector4, Matrix4, Vector2};
use core::f64::consts::PI;
use std::slice;
use std::fs::OpenOptions;
use std::io::prelude::*;
use std::str;
use splines::{Interpolation, Key, Spline};

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

fn runge_kutter_iteration(q: Vector4<f64>, h: f64, param: &Potential) -> Vector4<f64> {
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

struct Potential {
    de: f64,
    re: f64,
    a: f64
}

struct Atom {
    q_current: Vector4<f64>,
    mass: f64,
    x_history: Vec<f64>,
    y_history: Vec<f64>,
    vx_history: Vec<f64>,
    vy_history: Vec<f64>,
    current_it: usize,
    history: bool
}

impl Atom {
    fn add_record(&mut self, ind: usize) {
        self.x_history[ind] = self.q_current[0];
        self.y_history[ind] = self.q_current[1];
        self.vx_history[ind] = self.q_current[2];
        self.vy_history[ind] = self.q_current[3];
    }
    
    fn one_it(&mut self, h: f64, param: &Potential) {
        self.q_current = runge_kutter_iteration(self.q_current, h, param);
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

        writeln!(file, "{}", "time,x,y,v_x,v_y").unwrap();
        
        let mut t: f64 = 0.0;
        for i in 0..self.x_history.len() {
            let line = format!("{},{},{},{},{}", t, self.x_history[i],
                self.y_history[i], self.vx_history[i], self.vy_history[i]);
            writeln!(file, "{}", line).unwrap();
            t += h;
        }
    }
}

fn run_particle(init_x: Vector2<f64>, init_v: Vector2<f64>, h: f64, n: usize, 
        param: &Potential, record_atom: bool) -> Atom {
    let m = if record_atom {n/10} else {7};
    let mut he = Atom {
        q_current: Vector4::new(init_x[0], init_x[1], init_v[0], init_v[1]),
        mass: 1.0,
        x_history: vec![0.0; m],
        y_history: vec![0.0; m],
        vx_history: vec![0.0; m],
        vy_history: vec![0.0; m],
        current_it: 0,
        history: record_atom
    };
    he.add_record(0);
    
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

//fn interpolate_surf(xs: &[f64], ys: &[f64]) -> Spline<f64, f64> {
//
//}

//-----------------------------------------------------------------------------

#[no_mangle]
pub extern fn single_particle(x: f64, y: f64, vx: f64, vy: f64, h: f64, 
        n_it: u64, text: *const u8, len: u64, p1: *const f64) {
    let init_x = Vector2::new(x, y);
    let init_v = Vector2::new(vx, vy);
    
    // Have to go via C-string to get the file name
    assert!(!text.is_null());
    let c_str = unsafe { slice::from_raw_parts(text, len as usize) };
    let fname = str::from_utf8(&c_str).unwrap();
    
    // The parameters of the potential
    assert!(!p1.is_null());
    let p2 = unsafe { slice::from_raw_parts(p1, 3) };
    let param = Potential{
        de: p2[0],
        re: p2[1],
        a: p2[2]
    };
    
    let he_atom = run_particle(init_x, init_v, h, n_it as usize, &param, true);
    he_atom.write_trajectory(&fname.to_string(), h);
}

#[no_mangle]
pub extern fn multiple_particle(xs: *const f64, ys: *const f64, vxs: *const f64, 
        vys: *const f64, n_atom: u64, h: f64, n_it: usize, text: *const u8, len: u64, 
        p1: *const f64, record: u64) {
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
    
    // Data structures for all final positions and direction
    let mut final_pos =  vec![Vector2::new(0.0,0.0); n_atom as usize];
    let mut final_v = vec![Vector2::new(0.0,0.0); n_atom as usize];
    
    for i in 1..n {
        let record_atom = i % record as usize == 0;
        let init_x = Vector2::new(x[i], y[i]);
        let init_v = Vector2::new(vx[i], vy[i]);
        
        let he_atom = run_particle(init_x, init_v, h, n_it, &param, record_atom);
        // Only save some of the trajectories atoms
        if record_atom {
            let mut save_name = fname.to_string();
            save_name.push_str(&i.to_string());
            save_name.push_str(".csv");
            he_atom.write_trajectory(&save_name, h);
        }
        
        final_pos[i][0] = he_atom.q_current[0];
        final_pos[i][1] = he_atom.q_current[1];
        final_v[i][0] = he_atom.q_current[2];
        final_v[i][1] = he_atom.q_current[3];
    }
    
    // Save all final positions and directions"final_states.csv".as_string()
    save_pos_vel(&final_pos, &final_v, );
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
