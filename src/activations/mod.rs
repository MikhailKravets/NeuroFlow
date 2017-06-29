pub fn tanh(x: f64) -> f64{
    x.tanh()
}

pub fn der_tanh(x: f64) -> f64{
    1.0 - x.tanh().powi(2)
}