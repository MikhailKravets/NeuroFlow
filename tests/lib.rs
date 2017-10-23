extern crate nn_rust;
extern crate time;
extern crate rand;

use nn_rust::FeedForward;
use rand::distributions::IndependentSample;
use rand::distributions::range::Range;
use rand::distributions::normal::Normal;

use nn_rust::activators::tanh;
use nn_rust::estimators;


#[test]
fn xor(){
    let allowed_error = 0.08; // Max allowed error is 8%
    let mut nn = FeedForward::new(&[2, 2, 1]);
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
    let mut nn = FeedForward::new(&[2, 3, 4, 3]);
    let mut sample;
    let mut training_set: Vec<(Vec<f64>, Vec<f64>)> = Vec::new();
    let training_amount = (20f64 * estimators::widrows(&[3, 4, 3], 0.8)) as i32;

    let c1 = Normal::new(1f64, 0.5);
    let c2 = Normal::new(2f64, 1.0);
    let c3 = Normal::new(3f64, 0.35);

    let mut k = 0;
    for _ in 0..training_amount{
        if k == 0{
            training_set.push((vec![c1.ind_sample(&mut rand::thread_rng()), c1.ind_sample(&mut rand::thread_rng())],
                               vec![1f64, 0f64, 0f64]));
            k += 1;
        }
        else if k == 1 {
            training_set.push((vec![c2.ind_sample(&mut rand::thread_rng()), c2.ind_sample(&mut rand::thread_rng())],
                               vec![0f64, 1f64, 0f64]));
            k += 1;
        }
        else if k == 2 {
            training_set.push((vec![c3.ind_sample(&mut rand::thread_rng()), c3.ind_sample(&mut rand::thread_rng())],
                               vec![0f64, 0f64, 1f64]));
            k += 1;
        }
        else {
            k = 0;
        }
    }

    let rnd_range = Range::new(0, training_set.len());

    let prev = time::now_utc();
    nn.activation(nn_rust::Activator::Tanh);

    for _ in 0..50_000{
        k = rnd_range.ind_sample(&mut rand::thread_rng());
        nn.fit(&training_set[k].0, &training_set[k].1);
    }

    fn check(c: &[f64], class: usize) -> bool{
        let mut max = c[0];
        let mut max_i = 0;
        for i in 1..c.len(){
            if max < c[i]{
                max = c[i];
                max_i = i;
            }
        }
        max_i == class
    }

    {
        sample = [c1.ind_sample(&mut rand::thread_rng()), c1.ind_sample(&mut rand::thread_rng())];
        let res = nn.calc(&sample);
        println!("for: [{:?}], [1, 0, 0] -> {:?}", sample, res);
        assert!(check(&res, 0));
    }

    {
        sample = [c2.ind_sample(&mut rand::thread_rng()), c2.ind_sample(&mut rand::thread_rng())];
        let res = nn.calc(&sample);
        println!("for: [{:?}], [0, 1, 0] -> {:?}", sample, res);
        assert!(check(res, 1));
    }

    {
        sample = [c3.ind_sample(&mut rand::thread_rng()), c3.ind_sample(&mut rand::thread_rng())];
        let res = nn.calc(&sample);
        println!("for: [{:?}], [0, 0, 1] -> {:?}", sample, res);
        assert!(check(res, 2));
    }

    println!("\nSpend time: {}", (time::now_utc() - prev));
    assert!(true);
}

#[test]
#[ignore]
fn widrows(){
    let w = estimators::widrows(&[2, 1], 0.1);
    assert_eq!(w, 90f64);
}