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
    let mut nn = FeedForward::new(&[1, 8, 6, 1]);
    let mut data: DataSet = DataSet::new();
    let mut i = -3.0;

    while i <= 3.0 {
        data.push(&[i], &[i.sin()]);
        i += 0.1;
    }

    let prev = time::now_utc();

    nn.activation(Tanh)
        .learning_rate(0.05)
        .train(&data, 30_000);

    let mut res;

    i = 0.0;
    while i <= 0.3{
        res = nn.calc(&[i])[0];
        println!("for [{:.3}], [{:.3}] -> [{:.3}]", i, i.sin(), res);
        i += 0.05;
    }

    println!("\nSpend time: {:.3}", (time::now_utc() - prev).num_milliseconds() as f64 / 1000.0);
}
