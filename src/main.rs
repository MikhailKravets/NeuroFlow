mod network;

use network::NeuralNet;

fn main() {
    println!("Application starts!");

    let nn = NeuralNet::new(vec![2, 2, 1], 0.1, 0.1);

    for v in nn.layers{
        for val in v.y{
            println!("val: {}", val)
        }
        println!("----------")
    }

    println!("Application stops!");
}
