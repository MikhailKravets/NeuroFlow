mod network;

use network::MLP;

extern crate time;
extern crate rand;

use rand::distributions::IndependentSample;
use rand::distributions::range::Range;

fn main() {
    println!("Application starts!");

    let mut nn = MLP::new(vec![2, 1], 0.1, 0.1);
    let sc = &[
        (&[0f64, 0f64], &[0f64]),
        (&[1f64, 0f64], &[1f64]),
        (&[0f64, 1f64], &[1f64]),
        (&[1f64, 1f64], &[0f64]),
    ];
    let mut k;
    let rnd_range = Range::new(0, sc.len());
    let prev = time::now_utc();

    for _ in 0..50_000{
        k = rnd_range.ind_sample(&mut rand::thread_rng());
        nn.fit(sc[k].0, sc[k].1);
    }

    nn.print(network::Field::Weights);

    for v in sc{
        println!("Res for: [{}, {}], [{}] -> [{}]", v.0[0], v.0[1], v.1[0], nn.calc(v.0)[0]);
    }

    println!("\nSpend time: {}", (time::now_utc() - prev));
    println!("Application stops!");
}
