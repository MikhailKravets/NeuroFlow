mod network;

use network::NeuralNet;

fn main() {
    println!("Application starts!");

    let mut nn = NeuralNet::new(vec![2, 2, 1], 0.1, 0.1);

    nn.print(network::Type::Weights);

    nn.fit(&[0f64, 1f64], &[1f64]);

    println!("Application stops!");
}
