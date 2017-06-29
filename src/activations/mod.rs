fn tanh(x: f64) -> f64{
    x.tanh()
}

fn der_tanh(x: f64) -> f64{
    1.0 - x.tanh().powi(2)
}