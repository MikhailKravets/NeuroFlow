extern crate neuroflow;
extern crate time;
extern crate rand;

use neuroflow::FeedForward;
use neuroflow::data::{DataSet, Extractable};

use rand::distributions::IndependentSample;
use rand::distributions::range::Range;
use rand::distributions::normal::Normal;

use neuroflow::activators;
use neuroflow::activators::tanh;
use neuroflow::activators::Type::Tanh;
use neuroflow::activators::Type::Sigmoid;
use neuroflow::estimators;



fn main(){
    let allowed_error = 0.08; // Max allowed error is 8%
    let mut nn = FeedForward::new(&[1, 8, 6, 1]);
    nn.activation(Tanh).learning_rate(0.05);

    let mut k;
    let mut sc = Vec::new();

    let mut i: f64 = -3.0;
    while i <= 3.0 {
        sc.push((i, i.sin()));
        i += 0.1;
    }

    let rnd_range = Range::new(0, sc.len());
    let prev = time::now_utc();

    for _ in 0..30_000 {
        k = rnd_range.ind_sample(&mut rand::thread_rng());
        nn.fit(&[sc[k].0], &[sc[k].1]);
    }

    let mut res;

    i = -0.0;
    while i <= 2.0{
        res = nn.calc(&[i])[0];
        println!("for [{:.3}], [{:.3}] -> [{:.3}]",
                 i, i.sin(), res);

        //        if (res - i.sin()).abs() > allowed_error{
        //            assert!(false);
        //        }

        i += 0.05;
    }

    println!("\nSpend time: {:.5}", (time::now_utc() - prev));
}
