/// # IO NeuroFlow module
/// The module should contain functions and traits for saving
/// and restoring of neural networks from files, and buffers

use std::fs::File;
use std::io::Write;
use serde;
use bincode::{serialize, Infinite};

/// obj: it should be neural network to save.
/// I guess it should be generic type with necessary
/// implementation of some trait
pub fn save<T: serde::Serialize>(obj: &T, file_path: &str){
    let mut file = File::create(file_path).unwrap();
    let encoded: Vec<u8> = serialize(obj, Infinite).unwrap();

    file.write_all(&encoded);
    file.flush();
}

/// Function that load NN from file.
/// Is it possible to return generic type!?
pub fn load(file_path: &str){

}

/// Future function for saving in JSON string.
/// return: JSON string
pub fn to_json<T: serde::Serialize>(obj: T) {

}

/// Function for deserializing of JSON to NN struct
pub fn from_json(s: &str){

}