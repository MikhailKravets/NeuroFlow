extern crate neuroflow;

use neuroflow::data::DataSet;
use neuroflow::data::Extractable;


#[test]
fn test_data_set(){
    let mut ds: DataSet = DataSet::new();
    let (x, y) = (&[1.3, 4.7, 3.3], &[5.0, 4.5, 3.1]);
    ds.push(x, y);
    assert_eq!(x[0], ds.rand().0[0]);
    assert_eq!(x[0], ds.get(0).0[0]);
}

#[test]
#[ignore]
fn test_load_from_csv(){
    let ds = DataSet::from_csv("D:\\DELETE\\отчеты\\output.csv");
    match ds {
       Ok(v) => println!("{:?}", v),
        Err(e) => {
            println!("{}", e);
            assert!(false);
        }
    }
}