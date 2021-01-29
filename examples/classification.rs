extern crate neuroflow;
extern crate time;
extern crate rand;
extern crate rand_distr;

use neuroflow::FeedForward;

use rand::Rng;
use rand_distr::{Normal, Distribution};

use time::OffsetDateTime;

use neuroflow::estimators;


fn main(){
    //let allowed_error = 0.08; // Max allowed error is 8%
    let mut nn = FeedForward::new(&[2, 3, 4, 3]);
    let mut sample;
    let mut training_set: Vec<(Vec<f64>, Vec<f64>)> = Vec::new();
    let training_amount = (20f64 * estimators::widrows(&[3, 4, 3], 0.8)) as i32;

    let c1 = Normal::new(1f64, 0.5).unwrap();
    let c2 = Normal::new(2f64, 1.0).unwrap();
    let c3 = Normal::new(3f64, 0.35).unwrap();
    

    let mut k = 0;
    for _ in 0..training_amount{
        if k == 0{
            training_set.push((vec![c1.sample(&mut rand::thread_rng()), c1.sample(&mut rand::thread_rng())],
                               vec![1f64, 0f64, 0f64]));
            k += 1;
        }
            else if k == 1 {
                training_set.push((vec![c2.sample(&mut rand::thread_rng()), c2.sample(&mut rand::thread_rng())],
                                   vec![0f64, 1f64, 0f64]));
                k += 1;
            }
                else if k == 2 {
                    training_set.push((vec![c3.sample(&mut rand::thread_rng()), c3.sample(&mut rand::thread_rng())],
                                       vec![0f64, 0f64, 1f64]));
                    k += 1;
                }
                    else {
                        k = 0;
                    }
    }

    let mut rng = rand::thread_rng();    

    let prev = OffsetDateTime::now_utc();
    nn.activation(neuroflow::activators::Type::Tanh);

    for _ in 0..50_000{
        k = rng.gen_range(0..training_set.len());
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
        sample = [c1.sample(&mut rand::thread_rng()), c1.sample(&mut rand::thread_rng())];
        let res = nn.calc(&sample);
        println!("for: [{:?}], [1, 0, 0] -> {:?}", sample, res);
        assert!(check(&res, 0));
    }

    {
        sample = [c2.sample(&mut rand::thread_rng()), c2.sample(&mut rand::thread_rng())];
        let res = nn.calc(&sample);
        println!("for: [{:?}], [0, 1, 0] -> {:?}", sample, res);
        assert!(check(res, 1));
    }

    {
        sample = [c3.sample(&mut rand::thread_rng()), c3.sample(&mut rand::thread_rng())];
        let res = nn.calc(&sample);
        println!("for: [{:?}], [0, 0, 1] -> {:?}", sample, res);
        assert!(check(res, 2));
    }

    println!("\nSpend time: {}", (OffsetDateTime::now_utc() - prev).subsec_milliseconds());
    assert!(true);
}