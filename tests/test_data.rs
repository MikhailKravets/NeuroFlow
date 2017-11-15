extern crate neuroflow;

use neuroflow::data::DataSet;
use neuroflow::data::Extractable;

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
        file.write_all("x,x,y\n".as_bytes()).unwrap();
        file.write_all("0,0,0\n".as_bytes()).unwrap();
        file.write_all("1,0,1\n".as_bytes()).unwrap();
        file.write_all("0,1,1\n".as_bytes()).unwrap();
        file.write_all("1,1,0".as_bytes()).unwrap();
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