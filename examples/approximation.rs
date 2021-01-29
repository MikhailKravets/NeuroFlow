extern crate neuroflow;
extern crate time;
extern crate rand;
extern crate rand_distr;

use neuroflow::FeedForward;
use neuroflow::data::{DataSet};

use time::OffsetDateTime;

use neuroflow::activators::Type::Tanh;




fn main(){
    let mut nn = FeedForward::new(&[1, 7, 8, 8, 7, 1]);
    let mut data: DataSet = DataSet::new();
    let mut i = -3.0;

    while i <= 2.5 {
        data.push(&[i], &[0.5*(i.exp().sin()) - (-i.exp()).cos()]);
        i += 0.05;
    }

    let prev = OffsetDateTime::now_utc();

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

    println!("\nSpend time: {:.3}", (OffsetDateTime::now_utc() - prev).subsec_milliseconds() as f64 / 1000.0);
}
