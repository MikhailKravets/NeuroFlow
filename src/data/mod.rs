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
    y: Vec<Vec<f64>>,

    sum_x: Vec<f64>,
    sum_y: Vec<f64>,

    mean_x: Vec<f64>,
    mean_y: Vec<f64>,
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
        return DataSet{
            x: vec![],
            y: vec![],
            sum_x: vec![],
            sum_y: vec![],
            mean_x: vec![],
            mean_y: vec![]};
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

    /// Find sum of elements by columns in `DataSet`
    ///
    /// # Examples
    ///
    /// ```
    /// use neuroflow::data::DataSet;
    ///
    /// let mut data = DataSet::new();
    /// data.push(&[1.3], &[1.2, 2.1]);
    /// data.push(&[1.1], &[1.0, 2.0]);
    ///
    /// let (x, y) = data.sum();
    /// println!("{:?} {:?}", x, y);
    /// ```
    ///
    /// Expected output
    ///
    /// `[2.4] [2.2, 2.1]`
    pub fn sum(&mut self) -> (&[f64], &[f64]){
        while self.sum_x.len() < self.x.len(){
            self.sum_x.push(0.0);
        }
        for i in 0..self.x.len(){
            for j in 0..self.x[i].len(){
                self.sum_x[j] += self.x[i][j];
            }
        }

        while self.sum_y.len() < self.y.len(){
            self.sum_y.push(0.0);
        }
        for i in 0..self.y.len(){
            for j in 0..self.y[i].len(){
                self.sum_y[j] += self.y[i][j];
            }
        }

        (&self.sum_x, &self.sum_y)
    }

    /// Find mean value of each column in `DataSet`
    ///
    /// # Examples
    ///
    /// ```
    /// use neuroflow::data::DataSet;
    ///
    /// let mut data = DataSet::new();
    /// data.push(&[1.3], &[1.2, 2.1]);
    /// data.push(&[1.1], &[1.0, 2.0]);
    ///
    /// let (x, y) = data.mean();
    /// println!("{:?} {:?}", x, y);
    /// ```
    ///
    /// Expected output
    ///
    /// `[1.2] [1.1, 1.05]`
    pub fn mean(&mut self) -> (&[f64], &[f64]){
        self.sum();

        self.mean_x = self.sum_x.clone();
        for i in 0..self.mean_x.len(){
            self.mean_x[i] /= self.x.len() as f64;
        }

        self.mean_y = self.sum_y.clone();
        for i in 0..self.mean_y.len(){
            self.mean_y[i] /= self.y.len() as f64;
        }

        (&self.mean_x, &self.mean_y)
    }

    /// Round each value in `DataSet` with the given precision.
    ///
    /// * `precision: u32` - amount of digits after point that must be remained after rounding
    ///
    /// # Examples
    ///
    /// ```
    /// use neuroflow::data::DataSet;
    ///
    /// let mut data = DataSet::new();
    /// data.push(&[1.3456465], &[1.259898, 2.1113213]);
    /// data.push(&[1.11132132132], &[1.04848, 2.0548487]);
    ///
    /// data.round(2);
    /// ```
    pub fn round(&mut self, precision: u32){
        let pow = 10f64.powi(precision as i32);

        for i in 0..self.x.len(){
            for j in 0..self.x[i].len(){
                self.x[i][j] = (self.x[i][j] * pow).round() / pow;
            }
        }

        for i in 0..self.y.len(){
            for j in 0..self.y[i].len(){
                self.y[i][j] = (self.y[i][j] * pow).round() / pow;
            }
        }
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