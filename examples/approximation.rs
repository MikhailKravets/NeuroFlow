extern crate neuroflow;
extern crate time;
extern crate rand;

use neuroflow::FeedForward;
use neuroflow::data::DataSet;

use neuroflow::activators::Type::Tanh;



fn main(){
    let mut nn = FeedForward::new(&[1, 5, 3, 1]);
    let mut data: DataSet = DataSet::new();
    let mut i = -3.0;

    while i <= 2.5 {
        data.push(&[i], &[0.5*(i.exp().sin()) - (-i.exp()).cos()]);
        i += 0.05;
    }

    let prev = time::now_utc();

    nn.activation(Tanh)
        .learning_rate(0.007)
        .train(&data, 60_000);

    let mut res;

    i = 0.0;
    while i <= 0.3{
        res = nn.calc(&[i])[0];
        println!("for [{:.3}], [{:.3}] -> [{:.3}]", i, 0.5*(i.exp().sin()) - (-i.exp()).cos(), res);
        i += 0.07;
    }

    println!("\nSpend time: {:.3}", (time::now_utc() - prev).num_milliseconds() as f64 / 1000.0);
}
