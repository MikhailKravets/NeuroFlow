#[allow(dead_code)]
#[derive(Serialize, Deserialize)]
pub enum Type {
    Sigmoid,
    Tanh
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