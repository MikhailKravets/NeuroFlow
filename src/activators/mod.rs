//! Module contains popular neural networks activation functions
//! and theirs derivatives

use std::f64;

/// Determine types of activation functions contained in this module.
#[allow(dead_code)]
#[derive(Serialize, Deserialize)]
pub enum Type {
    Sigmoid,
    Tanh,
    Relu
}


pub fn sigm(x: f64) -> f64{ 1.0/(1.0 + x.exp()) }
pub fn der_sigm(x: f64) -> f64{
    sigm(x)*(1.0 - sigm(x))
}

pub fn tanh(x: f64) -> f64{
    x.tanh()
}

pub fn der_tanh(x: f64) -> f64{
    1.0 - x.tanh().powi(2)
}

pub fn relu(x: f64) -> f64{
    f64::max(0.0, x)
}
pub fn der_relu(x: f64) -> f64{
    if x <= 0.0 {
        0.0
    } else {
        1.0
    }
}
