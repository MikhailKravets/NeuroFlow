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
    let mut nn = FeedForward::new(&[1, 7, 8, 8, 7, 1]);
    let mut data: DataSet = DataSet::new();
    let mut i = -3.0;

    while i <= 2.5 {
        data.push(&[i], &[0.5*(i.exp().sin()) - (-i.exp()).cos()]);
        i += 0.05;
    }

    let prev = time::now_utc();

    nn.activation(Tanh)
        .learning_rate(0.01)
        .train(&data, 50_000);

    let mut res;

    i = 0.0;
    while i <= 0.3{
        res = nn.calc(&[i])[0];
        println!("for [{:.3}], [{:.3}] -> [{:.3}]", i, 0.5*(i.exp().sin()) - (-i.exp()).cos(), res);
        i += 0.07;
    }

    println!("\nSpend time: {:.3}", (time::now_utc() - prev).num_milliseconds() as f64 / 1000.0);
}
