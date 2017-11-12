/// TODO: write functions that loads csv data.
/// TODO: write functions that write existing NN to file.

use rand;
use rand::distributions::range::Range;
use rand::distributions::IndependentSample;

/// The function's prototype that loads data from csv
///  and place it in the DataSet structure
fn from_csv(file_path: &str) -> DataSet {
    DataSet::new()
}

pub trait Extractable {
    fn rand(&self) -> (&Vec<f64>, &Vec<f64>);
    fn get(&self, i: usize) -> (&Vec<f64>, &Vec<f64>);
    fn len(&self) -> usize;
}

pub struct DataSet{
    x: Vec<Vec<f64>>,
    y: Vec<Vec<f64>>
}

impl DataSet {
    pub fn new() -> DataSet{
        return DataSet{x: vec![], y: vec![]};
    }

    pub fn push(&mut self, x: &[f64], y: &[f64]){
        self.x.push(x.to_vec());
        self.y.push(y.to_vec());
    }

    pub fn remove(&mut self, i: usize){
        self.x.remove(i);
        self.y.remove(i);
    }
}

impl Extractable for DataSet{
    fn rand(&self) -> (&Vec<f64>, &Vec<f64>){
        let rnd_range = Range::new(0, self.y.len());
        let k = rnd_range.ind_sample(&mut rand::thread_rng());

        (&self.x[k], &self.y[k])
    }
    fn get(&self, i: usize) -> (&Vec<f64>, &Vec<f64>){
        (&self.x[i], &self.y[i])
    }
    fn len(&self) -> usize {
        self.y.len()
    }
}