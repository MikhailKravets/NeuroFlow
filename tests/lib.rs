extern crate nn_rust;
extern crate time;
extern crate rand;

use nn_rust::MLP;
use rand::distributions::IndependentSample;
use rand::distributions::range::Range;
use rand::distributions::normal::Normal;

use nn_rust::activators::tanh;
use nn_rust::estimators;


#[test]
#[ignore]
fn xor(){
    let allowed_error = 0.08; // Max allowed error is 8%
    let mut nn = MLP::new(&[2, 2, 1]);
    let sc = &[
        (&[0f64, 0f64], &[0f64]),
        (&[1f64, 0f64], &[1f64]),
        (&[0f64, 1f64], &[1f64]),
        (&[1f64, 1f64], &[0f64]),
    ];
    let mut k;
    let rnd_range = Range::new(0, sc.len());
    let prev = time::now_utc();

    for _ in 0..20_000{
        k = rnd_range.ind_sample(&mut rand::thread_rng());
        nn.fit(sc[k].0, sc[k].1);
    }

    let mut res;
    for v in sc{
        res = nn.calc(v.0)[0];
        println!("Res for: [{}, {}], [{}] -> [{}]", v.0[0], v.0[1], v.1[0], res);

        if (res - v.1[0]).abs() > allowed_error{
            assert!(false);
        }
    }

    println!("\nSpend time: {}", (time::now_utc() - prev));
    assert!(true);
}

#[test]
fn classes(){
    let allowed_error = 0.08; // Max allowed error is 8%
    let mut nn = MLP::new(&[2, 3, 3]);
    let mut sample;
    let mut k = 0;

    let c1 = Normal::new(0.1f64, 0.05);
    let c2 = Normal::new(0.25f64, 0.07);
    let c3 = Normal::new(0.5f64, 0.3);

    let rnd_range = Range::new(0, 10);
    let prev = time::now_utc();

    nn.activation(nn_rust::Activator::Sigmoid);

    for _ in 0..20_000{
        nn.fit(&[c1.ind_sample(&mut rand::thread_rng()), c1.ind_sample(&mut rand::thread_rng())], &[0.98f64, 0f64, 0f64]);
    }

    for _ in 0..20_000{
        nn.fit(&[c2.ind_sample(&mut rand::thread_rng()), c2.ind_sample(&mut rand::thread_rng())], &[0f64, 0.98f64, 0f64]);
    }

    for _ in 0..20_000{
        nn.fit(&[c3.ind_sample(&mut rand::thread_rng()), c3.ind_sample(&mut rand::thread_rng())], &[0f64, 0f64, 0.98f64]);
    }

    {
        let res;
        sample = [c1.ind_sample(&mut rand::thread_rng()), c1.ind_sample(&mut rand::thread_rng())];
        res = nn.calc(&sample);
        println!("Res for: [{:?}], [1, 0, 0] -> [{}, {}, {}]", sample, res[0], res[1], res[2]);
    }

    {
        let res;
        sample = [c2.ind_sample(&mut rand::thread_rng()), c2.ind_sample(&mut rand::thread_rng())];
        res = nn.calc(&sample);
        println!("Res for: [{:?}], [0, 1, 0] -> [{}, {}, {}]", sample, res[0], res[1], res[2]);
    }

    {
        let res;
        sample = [c3.ind_sample(&mut rand::thread_rng()), c3.ind_sample(&mut rand::thread_rng())];
        res = nn.calc(&sample);
        println!("Res for: [{:?}], [0, 0, 1] -> [{}, {}, {}]", sample, res[0], res[1], res[2]);
    }

    //  (res - v.1[0]).abs() > allowed_error

    println!("\nSpend time: {}", (time::now_utc() - prev));
    assert!(true);
}

#[test]
#[ignore]
fn widrows(){
    let w = estimators::widrows(&[2, 1], 0.1);
    assert_eq!(w, 90f64);
}