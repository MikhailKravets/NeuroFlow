//! The module contains functions, structs, enums, and traits
//! for input/output neural networks. E. g. it can save network
//! to the file and then loads it back.
//!
//! # Example
//! Saving of neural network:
//!
//! ```
//! use neuroflow::FeedForward;
//! use neuroflow::io;
//!
//! let mut nn = FeedForward::new(&[2, 2, 1]);
//! /* train here your neural network */
//! io::save(&nn, "test.flow");
//! ```
//!
//! Restoring of neural network:
//!
//! ```
//! use neuroflow::FeedForward;
//! use neuroflow::io;
//!
//! let mut new_nn: FeedForward = io::load("test.nn")
//!     .unwrap_or(FeedForward::new(&[2, 2, 1]));
//! ```

use std;
use std::fs::File;
use std::io::{Write, BufReader};
use serde;
use bincode;
use bincode::{serialize, deserialize_from, Infinite};

use ErrorKind;

/// Saves given neural network to file specified by `file_path`.
///
/// * `obj: &T` - link on neural network;
/// * `file_path: &str` - path to the file.
/// * `return -> Result<(), IOError>` - result of operation;
///
/// # Examples
///
/// ```
/// use neuroflow::FeedForward;
/// use neuroflow::io;
///
/// let mut nn = FeedForward::new(&[2, 2, 1]);
/// /* train here your neural network */
/// io::save(&nn, "test.flow");
/// ```
pub fn save<T: serde::Serialize>(obj: &T, file_path: &str) -> Result<(), ErrorKind>{
    let mut file = File::create(file_path).map_err(ErrorKind::IO)?;
    let encoded: Vec<u8> = serialize(obj, Infinite).map_err(ErrorKind::Encoding)?;

    file.write_all(&encoded).map_err(ErrorKind::IO)?;

    Ok(())
}

/// Loads and restores the neural network from file.
///
/// * `file_path: &'b str` - path to the file;
/// * `return -> Result<T, IOError>` - if Ok returns loaded neural network (Note, you must
/// apparently specify the type T).
///
/// # Examples
///
/// ```
/// use neuroflow::FeedForward;
/// use neuroflow::io;
///
/// let mut new_nn: FeedForward = io::load("test.flow")
///     .unwrap_or(FeedForward::new(&[2, 2, 1]));
/// ```
pub fn load<'b, T>(file_path: &'b str) -> Result<T, ErrorKind> where for<'de> T: serde::Deserialize<'de>{
    let file = File::open(file_path).map_err(ErrorKind::IO)?;
    let mut buf = BufReader::new(file);

    let mut nn: T = deserialize_from(&mut buf, Infinite).map_err(ErrorKind::Encoding)?;

    Ok(nn)
}

/// Future function for saving in JSON string.
/// return: JSON string
pub fn to_json<T: serde::Serialize>(obj: T) {

}

/// Function for deserializing of JSON to NN struct
pub fn from_json(s: &str){

}