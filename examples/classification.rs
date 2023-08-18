extern crate neuroflow;
extern crate time;
extern crate rand;
extern crate rand_distr;

use neuroflow::FeedForward;

use rand::distributions::Uniform;
use rand_distr::Normal;

use neuroflow::estimators;
use rand::{thread_rng, Rng};


fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut nn = FeedForward::new(&[2, 3, 4, 3]);
    let mut sample;
    let mut training_set: Vec<(Vec<f64>, Vec<f64>)> = Vec::new();
    let training_amount = (20f64 * estimators::widrows(&[3, 4, 3], 0.8)) as i32;

    let mut rng = thread_rng();
    let c1 = Normal::new(1f64, 0.5)?;
    let c2 = Normal::new(2f64, 1.0)?;
    let c3 = Normal::new(3f64, 0.35)?;

    let mut k = 0;
    for _ in 0..training_amount{
        if k == 0{
            training_set.push((vec![rng.sample(c1), rng.sample(c1)], vec![1f64, 0f64, 0f64]));
            k += 1;
        }
            else if k == 1 {
                training_set.push((vec![rng.sample(c2), rng.sample(c2)], vec![0f64, 1f64, 0f64]));
                k += 1;
            }
                else if k == 2 {
                    training_set.push((vec![rng.sample(c3), rng.sample(c3)], vec![0f64, 0f64, 1f64]));
                    k += 1;
                }
                    else {
                        k = 0;
                    }
    }

    let rnd_range = Uniform::new(0, training_set.len());

    let prev = time::now_utc();
    nn.activation(neuroflow::activators::Type::Tanh);

    for _ in 0..50_000{
        k = rng.sample(rnd_range);
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
        sample = [rng.sample(c1), rng.sample(c1)];
        let res = nn.calc(&sample);
        println!("for: [{:?}], [1, 0, 0] -> {:?}", sample, res);
        assert!(check(&res, 0));
    }

    {
        sample = [rng.sample(c2), rng.sample(c2)];
        let res = nn.calc(&sample);
        println!("for: [{:?}], [0, 1, 0] -> {:?}", sample, res);
        assert!(check(&res, 1));
    }

    {
        sample = [rng.sample(c3), rng.sample(c3)];
        let res = nn.calc(&sample);
        println!("for: [{:?}], [0, 0, 1] -> {:?}", sample, res);
        assert!(check(&res, 2));
    }

    println!("\nSpend time: {}", (time::now_utc() - prev));
    assert!(true);

    Ok(())
}