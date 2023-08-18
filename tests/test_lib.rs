extern crate neuroflow;
extern crate time;
extern crate rand;

use neuroflow::FeedForward;
use neuroflow::data::{DataSet, Extractable};

use rand::{thread_rng, Rng};
use rand::distributions::Uniform;

use neuroflow::activators;
use neuroflow::estimators;


#[test]
fn xor(){
    const ALLOWED_ERROR: f64 = 0.1; // Max allowed error is 10%
    let mut nn = FeedForward::new(&[2, 4, 1]);
    let sc = &[
        (&[0f64, 0f64], &[0f64]),
        (&[1f64, 0f64], &[1f64]),
        (&[0f64, 1f64], &[1f64]),
        (&[1f64, 1f64], &[0f64]),
    ];
    let prev = time::now_utc();

    let mut k;
    let mut rnd_range = thread_rng();

    nn.learning_rate(0.1).momentum(0.01);
    for _ in 0..30_000{
        k = rnd_range.sample(Uniform::new(0, sc.len()));
        nn.fit(sc[k].0, sc[k].1);
    }

    let mut res;
    for v in sc{
        res = nn.calc(v.0)[0];
        println!("for [{:.3}, {:.3}], [{:.3}] -> [{:.3}]",
                 v.0[0], v.0[1], v.1[0], res);

        if (res - v.1[0]).abs() > ALLOWED_ERROR {
            assert!(false);
        }
    }

    println!("\nSpend time: {:.5}", (time::now_utc() - prev));
    assert!(true);
}

#[test]
fn xor_through_data_set_and_train(){
    const ALLOWED_ERROR: f64 = 0.1; // Max allowed error is 10%
    let mut nn = FeedForward::new(&[2, 4, 1]);
    let mut data = DataSet::new();

    data.push(&[0f64, 0f64], &[0f64]);
    data.push(&[1f64, 0f64], &[1f64]);
    data.push(&[0f64, 1f64], &[1f64]);
    data.push(&[1f64, 1f64], &[0f64]);

    nn.activation(activators::Type::Tanh)
        .learning_rate(0.01)
        .momentum(0.1)
        .train(&data, 30_000);

    let mut res;
    let mut d;
    for i in 0..data.len(){
        res = nn.calc(data.get(i).0)[0];
        d = data.get(i);
        println!("for [{:.3}, {:.3}], [{:.3}] -> [{:.3}]", d.0[0], d.0[1], d.1[0], res);
        if (res - data.get(i).1[0]).abs() > ALLOWED_ERROR {
            assert!(false);
        }
    }
}

#[test]
fn binding(){
    let mut nn = FeedForward::new(&[6, 4, 4, 2, 1]);

    println!("{}", nn);

    nn.unbind(1, 0);
    println!("{}", nn);

    nn.bind(1, 0);
    println!("{}", nn);
}

#[test]
fn custom_activation(){
    fn func(_x: f64) -> f64{
        0.0
    }

    fn der_func(_x: f64) -> f64{
        0.0
    }

    let mut nn = FeedForward::new(&[1, 2, 1]);
    nn.custom_activation(func, der_func);

    let before_fit: f64 = nn.calc(&[3.2])[0];

    nn.fit(&[1.0], &[2.1]);

    let after_fit: f64 = nn.calc(&[2.1])[0];
    assert_eq!(before_fit, after_fit);
}

#[test]
fn widrows(){
    let w = estimators::widrows(&[2, 1], 0.1);
    assert_eq!(w, 90f64);
}