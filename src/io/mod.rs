/// # IO NeuroFlow module
/// The module should contain functions and traits for saving
/// and restoring of neural networks from files, and buffers

extern crate serde;

use std::fs::File;

/// obj: it should be neural network to save.
/// I guess it should be generic type with necessary
/// implementation of some trait
fn save<T: serde::Serialize>(obj: T, file_path: &str){
    let mut file = File::create(file_path).unwrap();

}

/// Function that load NN from file.
/// Is it possible to return generic type!?
fn load(file_path: &str){

}

/// Future function for saving in JSON string.
/// return: JSON string
fn to_json<T: serde::Serialize>(obj: T) {

}

/// Function for deserializing of JSON to NN struct
fn from_json(s: &str){

}