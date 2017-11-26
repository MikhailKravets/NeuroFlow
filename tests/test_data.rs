extern crate neuroflow;

use neuroflow::FeedForward;
use neuroflow::data::DataSet;
use neuroflow::data::Extractable;

use neuroflow::activators;

use std::io::Write;


#[test]
fn test_data_set(){
    let mut ds: DataSet = DataSet::new();
    let (x, y) = (&[1.3, 4.7, 3.3], &[5.0, 4.5, 3.1]);
    ds.push(x, y);
    assert_eq!(x[0], ds.rand().0[0]);
    assert_eq!(x[0], ds.get(0).0[0]);
}

#[test]
fn test_load_from_csv(){
    {
        let mut file = std::fs::File::create("test.csv").unwrap();
        file.write_all("0,0,-,0\n".as_bytes()).unwrap();
        file.write_all("1,0,-,1\n".as_bytes()).unwrap();
        file.write_all("0,1,-,1\n".as_bytes()).unwrap();
        file.write_all("1,1,-,0".as_bytes()).unwrap();
        file.flush().unwrap();
    }
    let ds = DataSet::from_csv("test.csv");

    std::fs::remove_file("test.csv").unwrap();
    match ds {
       Ok(v) => println!("{:?}", v),
        Err(e) => {
            println!("{}", e);
            assert!(false);
        }
    }
}

#[test]
fn test_sum(){
    let mut data = DataSet::new();

    data.push(&[0f64, 0f64], &[0f64]);
    data.push(&[1f64, 0f64], &[1f64]);
    data.push(&[0f64, 1f64], &[1f64]);
    data.push(&[1f64, 1f64], &[0f64]);

    let (x, y) = data.sum();

    assert_eq!(x[0], 2.0);
    assert_eq!(x[1], 2.0);
    assert_eq!(y[0], 2.0);
}

#[test]
fn test_mean(){
    let mut data = DataSet::new();

    data.push(&[0f64, 0f64], &[0f64]);
    data.push(&[1f64, 0f64], &[1f64]);
    data.push(&[0f64, 1f64], &[1f64]);
    data.push(&[1f64, 1f64], &[0f64]);

    let (x, y) = data.mean();

    assert_eq!(x[0], 0.5);
    assert_eq!(x[1], 0.5);
    assert_eq!(y[0], 0.5);
}

#[test]
fn test_round(){
    use neuroflow::data::Extractable;

    let mut data = DataSet::new();
    data.push(&[0.54878, 0.124578], &[0.12357]);
    data.push(&[1.9879849, 0.45646546], &[1.98798745]);
    data.push(&[0.78798789, 1.9798798], &[1.3248778]);
    data.push(&[1.98798798, 1.98789456], &[0.97878945]);

    data.round(2);

    assert_eq!(data.get(0).0[0], 0.55);
    assert_eq!(data.get(0).1[0], 0.12);

    println!("{:?}", data);
}

#[test]
fn test_division(){
    use neuroflow::data::Extractable;

    let mut data = DataSet::new();
    data.push(&[0.54878, 0.124578], &[0.12357]);
    data.push(&[1.9879849, 0.45646546], &[1.98798745]);
    data.push(&[0.78798789, 1.9798798], &[1.3248778]);
    data.push(&[1.98798798, 1.98789456], &[0.97878945]);
    data.push(&[1.98798798, 1.98789456], &[0.97878945]);
    data.push(&[1.98798798, 1.98789456], &[0.97878945]);
    data.push(&[1.98798798, 1.98789456], &[0.97878945]);
    data.push(&[1.98798798, 1.98789456], &[0.97878945]);
    data.push(&[1.98798798, 1.98789456], &[0.97878945]);
    data.push(&[1.98798798, 1.98789456], &[0.97878945]);

    data.round(2);

    data.divide(0.1);
    assert_eq!(data.len(), 9);

    data.divide(0.1);
    assert_eq!(data.len(), 9);
}

#[test]
fn test_cv(){
    const ALLOWED_ERROR: f64 = 0.1; // Max allowed error is 10%
    let mut nn = FeedForward::new(&[2, 2, 2]);
    let mut data = DataSet::new();

    data.push(&[0f64, 0f64], &[0f64, 0f64]);
    data.push(&[1f64, 0f64], &[1f64, 0f64]);
    data.push(&[0f64, 1f64], &[1f64, 0f64]);
    data.push(&[1f64, 1f64], &[0f64, 0f64]);
    data.push(&[0f64, 2f64], &[1f64, 0f64]);
    data.push(&[1f64, 2f64], &[2f64, 0f64]);
    data.push(&[2f64, 1f64], &[1f64, 0f64]);

    data.divide(0.5);

    nn.activation(activators::Type::Tanh)
        .learning_rate(0.05)
        .momentum(0.15)
        .train(&data, 30_000);

    println!("{:?}", data.cv(&mut nn));
}