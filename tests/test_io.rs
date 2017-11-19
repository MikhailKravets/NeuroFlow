extern crate neuroflow;
extern crate time;
extern crate rand;

use std::path::Path;
use std::fs::remove_file;

use neuroflow::FeedForward;
use neuroflow::data::{DataSet};

use neuroflow::activators;

use neuroflow::io::{save, load, to_json};


#[test]
fn saving_to_json(){
    let mut nn = FeedForward::new(&[2, 2, 1]);
    let mut data = DataSet::new();

    data.push(&[0f64, 0f64], &[0f64]);
    data.push(&[1f64, 0f64], &[1f64]);
    data.push(&[0f64, 1f64], &[1f64]);
    data.push(&[1f64, 1f64], &[0f64]);

    nn.activation(activators::Type::Tanh)
        .learning_rate(0.05)
        .momentum(0.15)
        .train(&data, 5_000);

    match to_json(&nn) {
        Ok(s) => println!("{}", s),
        Err(e) => {
            println!("{:?}", e);
            assert!(false);
        }
    };
}

#[test]
fn saving_of_neural_net(){
    let mut nn = FeedForward::new(&[2, 2, 1]);
    let mut data = DataSet::new();

    data.push(&[0f64, 0f64], &[0f64]);
    data.push(&[1f64, 0f64], &[1f64]);
    data.push(&[0f64, 1f64], &[1f64]);
    data.push(&[1f64, 1f64], &[0f64]);

    nn.activation(activators::Type::Tanh)
        .learning_rate(0.05)
        .momentum(0.15)
        .train(&data, 5_000);

    save(&nn, "test.nn").unwrap();

    let p = Path::new("test.nn");
    assert!(p.exists());
    if p.exists(){
        remove_file(p).unwrap();
    }
}

#[test]
fn loading_of_neural_net(){
    let mut nn = FeedForward::new(&[2, 2, 1]);
    let mut data = DataSet::new();

    data.push(&[0f64, 0f64], &[0f64]);
    data.push(&[1f64, 0f64], &[1f64]);
    data.push(&[0f64, 1f64], &[1f64]);
    data.push(&[1f64, 1f64], &[0f64]);

    nn.activation(activators::Type::Tanh)
        .learning_rate(0.05)
        .momentum(0.15)
        .train(&data, 5_000);

    save(&nn, "test.nn").unwrap();

    let mut new_nn = load::<FeedForward>("test.nn").unwrap();

    let sc = &[
        (&[0f64, 0f64], &[0f64]),
        (&[1f64, 0f64], &[1f64]),
        (&[0f64, 1f64], &[1f64]),
        (&[1f64, 1f64], &[0f64]),
    ];

    let mut res;
    let mut res1;
    for v in sc{
        res = nn.calc(v.0)[0];
        res1 = new_nn.calc(v.0)[0];
        println!("for [{:.3}, {:.3}] -> [{:.3}], [{:.3}]",
                 v.0[0], v.0[1], res, res1);

        if (res - res1).abs() > 0.1{
            assert!(false);
        }
    }

    let p = Path::new("test.nn");
    remove_file(p).unwrap();
}

#[test]
fn load_not_existent_file(){
    match load::<FeedForward>("test.nn") {
        Ok(_) => assert!(false),
        Err(_) => assert!(true)
    }
}