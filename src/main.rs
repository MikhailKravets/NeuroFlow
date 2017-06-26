mod network;

use network::NeuralNet;

fn main() {
    println!("Application starts!");

    let nn = NeuralNet::new(vec![2, 2, 1], 0.1, 0.1);

    nn.print(network::Type::Weights);

    println!("Application stops!");
}
