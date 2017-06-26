mod network;

use network::NeuralNet;

fn main() {
    println!("Application starts!");

    let mut nn = NeuralNet::new(vec![2, 2, 1], 0.1, 0.1);

    let sc = &[
        (&[0f64, 0f64], &[0f64]),
        (&[1f64, 0f64], &[1f64]),
        (&[0f64, 1f64], &[1f64]),
        (&[1f64, 1f64], &[0f64]),
    ];
    let mut k = 0;

    for _ in 0..100{
        if k == 4{
            k = 0;
        }
        nn.fit(sc[k].0, sc[k].1);
        k += 1;
    }

    nn.print(network::Type::Deltas);

    println!("Application stops!");
}
