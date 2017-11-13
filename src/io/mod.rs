/// # IO NeuroFlow module
/// The module should contain functions and traits for saving
/// and restoring of neural networks from files, and buffers

use std;
use std::fs::File;
use std::io::{Read, Write, BufReader};
use serde;
use bincode;
use bincode::{serialize, deserialize_from, Infinite};

#[derive(Debug)]
pub enum IOError{
    IO(std::io::Error),
    Encoding(bincode::Error)
}


pub fn save<T: serde::Serialize>(obj: &T, file_path: &str) -> Result<(), IOError>{
    let mut file = File::create(file_path).map_err(IOError::IO)?;
    let encoded: Vec<u8> = serialize(obj, Infinite).map_err(IOError::Encoding)?;

    file.write_all(&encoded);
    file.flush();

    Ok(())
}


pub fn load<'b, T>(file_path: &'b str) -> Result<T, IOError> where for<'de> T: serde::Deserialize<'de>{
    let mut content: Vec<u8> = Vec::new();
    let mut file = File::open(file_path).map_err(IOError::IO)?;
    let mut buf = BufReader::new(file);

    let mut nn: T = deserialize_from(&mut buf, Infinite).map_err(IOError::Encoding)?;

    Ok(nn)
}

/// Future function for saving in JSON string.
/// return: JSON string
pub fn to_json<T: serde::Serialize>(obj: T) {

}

/// Function for deserializing of JSON to NN struct
pub fn from_json(s: &str){

}