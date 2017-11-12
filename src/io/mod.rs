/// # IO NeuroFlow module
/// The module should contain functions and traits for saving
/// and restoring of neural networks from files, and buffers



/// obj: it should be neural network to save.
/// I guess it should be generic type with necessary
/// implementation of some trait
fn save(obj: None, file_path: &str){

}

/// Function that load NN from file.
/// Is it possible to return generic type!?
fn load<T>(file_path: &str) -> None{

}

/// Future function for saving in JSON string.
/// return: JSON string
fn to_json(obj: None) -> &str {

}

/// Function for deserializing of JSON to NN struct
fn from_json(s: &str){

}