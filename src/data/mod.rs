//! Module contains functions, structs and traits for data storage, access, and processing.
//!
//! # Examples
//! In order to train network by `neuroflow::FeedForward::train` method,
//! the first argument of this method must implement Extractable trait.
//! Example for DataSet
//! ```text, no_run
//! impl Extractable for DataSet{
//!     fn rand(&self) -> (&Vec<f64>, &Vec<f64>){
//!         let rnd_range = Range::new(0, self.y.len());
//!         let k = rnd_range.ind_sample(&mut rand::thread_rng());
//!
//!         (&self.x[k], &self.y[k])
//!     }
//!     fn get(&self, i: usize) -> (&Vec<f64>, &Vec<f64>){
//!         (&self.x[i], &self.y[i])
//!     }
//!     fn len(&self) -> usize {
//!         self.y.len()
//!     }
//! }
//! ```
//!
//! Also, it has `DataSet` struct for easy managing of data.
//! For example, when you load data from file, it'll be placed into `DataSet`
//! ```text, no_run
//! // under development :(
//! ```
use rand;
use rand::distributions::range::Range;
use rand::distributions::IndependentSample;


fn from_csv(file_path: &str) -> DataSet {
    unimplemented!();
}

/// Trait for getting specific element from set.
///
/// # Example
/// ```text, no_run
/// impl Extractable for DataSet{
///     fn rand(&self) -> (&Vec<f64>, &Vec<f64>){
///        let rnd_range = Range::new(0, self.y.len());
///         let k = rnd_range.ind_sample(&mut rand::thread_rng());
///
///         (&self.x[k], &self.y[k])
///     }
///     fn get(&self, i: usize) -> (&Vec<f64>, &Vec<f64>){
///         (&self.x[i], &self.y[i])
///     }
///     fn len(&self) -> usize {
///         self.y.len()
///     }
/// }
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

/// Container for data set.
///
pub struct DataSet{
    x: Vec<Vec<f64>>,
    y: Vec<Vec<f64>>
}

impl DataSet {
    /// `DataSet` constructor.
    ///
    /// # Example
    /// ```text, no_run
    /// let mut data = DataSet::new();
    /// ```
    pub fn new() -> DataSet{
        return DataSet{x: vec![], y: vec![]};
    }

    /// Append data to the end of the set.
    ///
    /// * `x: &[f64]` - input data to neural network;
    /// * `y: &[f64]` - expected output of neural network.
    ///
    /// # Example
    /// ```text, no_run
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
    /// # Example
    /// ```text, no_run
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