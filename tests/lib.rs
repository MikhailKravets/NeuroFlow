extern crate nn_rust;
extern crate time;
extern crate rand;

use nn_rust::MLP;
use rand::distributions::IndependentSample;
use rand::distributions::range::Range;

use nn_rust::activators::tanh;
use nn_rust::estimators;


#[test]
#[ignore]
fn xor(){
    println!("Application starts!");

    let allowed_error = 0.08; // Max allowed error is 8%
    let mut nn = MLP::new(vec![2, 1]);
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
    println!("Application stops!");
    assert!(true);
}

#[test]
fn widrows(){
    let w = estimators::widrows(&[2, 1], 0.1);
    assert_eq!(w, 90f64);
}