//! Module contains functions, structs and traits for data storage, access, and processing.
//!
//! In order to train network by `neuroflow::FeedForward::train` method,
//! you need the first argument to implement `Extractable` trait.
//!
//! Also, it has `DataSet` struct (which implement `Executable` trait) for easy managing of data.
//! For example, when you load data from file, it'll be placed into `DataSet`.
use std;

use rand;
use rand::distributions::range::Range;
use rand::distributions::IndependentSample;
use csv;

use ErrorKind;

/// Trait for getting specific element from set.
///
/// # Examples
///
/// ```
/// use neuroflow::data::Extractable;
/// use neuroflow::data::DataSet;
///
/// let mut data = DataSet::new();
/// data.push(&[3.01], &[4.0]);
/// data.rand();
/// ```
pub trait Extractable {
    /// Get random element from set
    ///
    /// * `return` - tuple of two links on vectors.
    fn rand(&self) -> (&Vec<f64>, &Vec<f64>);

    /// Get element from set by index
    ///
    /// * `i: usize` - index of element;
    /// * `return` - tuple of two links on vectors.
    fn get(&self, i: usize) -> (&Vec<f64>, &Vec<f64>);

    /// Get length of set
    ///
    /// * `return` - length of set.
    fn len(&self) -> usize;
}

/// Container for the set of data.
#[derive(Debug)]
pub struct DataSet{
    x: Vec<Vec<f64>>,
    y: Vec<Vec<f64>>
}

impl DataSet {
    /// `DataSet` constructor.
    ///
    /// # Examples
    ///
    /// ```
    /// use neuroflow::data::DataSet;
    ///
    /// let mut data = DataSet::new();
    /// ```
    pub fn new() -> DataSet{
        return DataSet{x: vec![], y: vec![]};
    }

    /// Read data from csv file and parse it to the `DataSet` instance.
    ///
    /// The file must not have header. Input vector must be separated from desired output
    /// by `-` symbol like in the following:
    ///
    /// `1,0,1,-,1,2`
    ///
    /// `2,3,0,-,2,3,1`
    ///
    /// * `file_path: &str` - path to `csv` file;
    /// * `return -> Result<DataSet, Box<std::error::Error>>` - return new `DataSet`
    /// instance if Ok.
    ///
    /// # Examples
    ///
    /// ```
    /// use neuroflow::data::DataSet;
    ///
    /// let mut data = DataSet::from_csv("container.csv");
    /// println!("{:?}", data);
    /// ```
    pub fn from_csv(file_path: &str) -> Result<DataSet, Box<std::error::Error>> {
        let mut file = csv::ReaderBuilder::new()
            .has_headers(false)
            .from_path(file_path)?;
        let mut data_set = DataSet::new();
        let mut is_x = true;

        for row in file.records(){
            let records = row?;
            let mut x: Vec<f64> = Vec::new();
            let mut y: Vec<f64> = Vec::new();

            is_x = true;

            for i in 0..records.len(){
                if records.get(i).unwrap() == "-"{
                    is_x = false;
                    continue;
                } else if let Some(v) = records.get(i){
                    if is_x {
                        x.push(v.parse()?);
                    } else {
                        y.push(v.parse()?);
                    }
                }
            }
            data_set.push(&x, &y);
        }

        Ok(data_set)
    }

    /// Append data to the end of the set.
    ///
    /// * `x: &[f64]` - input data to neural network;
    /// * `y: &[f64]` - expected output of neural network.
    ///
    /// # Examples
    ///
    /// ```
    /// use neuroflow::data::DataSet;
    ///
    /// let mut data = DataSet::new();
    /// data.push(&[1.3], &[1.2, 2.1]);
    /// ```
    pub fn push(&mut self, x: &[f64], y: &[f64]){
        self.x.push(x.to_vec());
        self.y.push(y.to_vec());
    }

    /// Remove element by index from set
    ///
    /// * `i: usize` - index of element to be deleted.
    ///
    /// # Examples
    ///
    /// ```
    /// use neuroflow::data::DataSet;
    ///
    /// let mut data = DataSet::new();
    /// data.push(&[1.3], &[1.2, 2.1]);
    /// data.remove(0);
    /// ```
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