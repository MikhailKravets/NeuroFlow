


fn act(x: f64) -> f64{
    x.tanh()
}

fn der_act(x: f64) -> f64{
    1.0 - x.tanh().powi(2)
}


struct NeuralLayer{
    y: Vec<f64>,
    delta: Vec<f64>,
    w: Vec<Vec<f64>>,
}


struct NeuralNet{
    layers: Vec<NeuralLayer>,
    learn_rate: f32,
    moment: f32
}

impl NeuralLayer{

}

impl NeuralNet{
    fn new(&self, architecture: Vec<int32>, l_rate: f32, moment: f32) -> NeuralNet {
        for v in architecture{
            // create layers with v neurons
        }
    }
}


fn main() {
    println!("Application starts!");

    /* The final code here */

    println!("Application stops!");
}
