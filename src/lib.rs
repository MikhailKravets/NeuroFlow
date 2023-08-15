//! NeuroFlow is neural networks (and deep learning of course) Rust crate.
//! It relies on three pillars: speed, reliability, and speed again.
//!
//! Let's better check some examples.
//!
//! # Examples
//!
//! Here we are going to approximate very simple function `0.5*sin(e^x) - cos(e^(-x))`.
//!
//! ```rust
//!
//! use neuroflow::FeedForward;
//! use neuroflow::data::DataSet;
//! use neuroflow::activators::Type::Tanh;
//!
//!
//!  /*
//!      Define neural network with 1 neuron in input layers. Network contains 4 hidden layers.
//!      And, such as our function returns single value, it is reasonable to have 1 neuron in
//!      the output layer.
//!  */
//!  let mut nn = FeedForward::new(&[1, 7, 8, 8, 7, 1]);
//!
//!  /*
//!      Define DataSet.
//!
//!      DataSet is the Type that significantly simplifies work with neural network.
//!      Majority of its functionality is still under development :(
//!  */
//!  let mut data: DataSet = DataSet::new();
//!  let mut i = -3.0;
//!
//!  // Push the data to DataSet (method push accepts two slices: input data and expected output)
//!  while i <= 2.5 {
//!      data.push(&[i], &[0.5*(i.exp().sin()) - (-i.exp()).cos()]);
//!      i += 0.05;
//!  }
//!
//!  // Here, we set necessary parameters and train neural network
//!  // by our DataSet with 50 000 iterations
//!  nn.activation(Tanh)
//!      .learning_rate(0.01)
//!      .train(&data, 50_000);
//!
//!  let mut res;
//!
//!  // Let's check the result
//!  i = 0.0;
//!  while i <= 0.3{
//!      res = nn.calc(&[i])[0];
//!      println!("for [{:.3}], [{:.3}] -> [{:.3}]", i, 0.5*(i.exp().sin()) - (-i.exp()).cos(), res);
//!      i += 0.07;
//!  }
//! ```
//!
//! You don't need to lose your so hardly trained network, my friend! For those there are
//! functions for saving and loading of neural networks to and from file. They are
//! located in the `neuroflow::io` module.
//!
//! ```rust
//! # use neuroflow::FeedForward;
//! use neuroflow::io;
//! # let mut nn = FeedForward::new(&[1, 7, 8, 8, 7, 1]);
//!  /*
//!     In order to save neural network into file call function save from neuroflow::io module.
//!
//!     First argument is link on the saving neural network;
//!     Second argument is path to the file.
//! */
//! io::save(&mut nn, "test.flow").unwrap();
//!
//! /*
//!     After we have saved the neural network to the file we can restore it by calling
//!     of load function from neuroflow::io module.
//!
//!     We must specify the type of new_nn variable.
//!     The only argument of load function is the path to file containing
//!     the neural network
//! */
//! let mut new_nn: FeedForward = io::load("test.flow").unwrap();
//! ```
//!
//! We did say a little words about `DataSet` structure. It deserves to be considered
//! more precisely.
//!
//! Simply saying `DataSet` is just container for your input vectors and desired output to them,
//! but with additional functionality.
//!
//! ```rust
//! use std::path::Path;
//! use neuroflow::data::DataSet;
//!
//! // You can create empty DataSet calling its constructor new
//! let mut d1 = DataSet::new();
//!
//! // To push new data to DataSet instance call push method
//! d1.push(&[0.1, 0.2], &[1.0, 2.3]);
//! d1.push(&[0.05, 0.01], &[0.5, 1.1]);
//!
//! // You can load data from csv file
//! let p = "file.csv";
//! if Path::new(p).exists(){
//!     let mut d2 = DataSet::from_csv(p); // Easy, eah?
//! }
//!
//! // You can round all DataSet elements with precision
//! d1.round(2); // 2 is the amount of digits after point
//!
//! // Also, it is possible to get some statistical information.
//! // For current version it is possible to get only mean values (by each dimension or by
//! // other words each column in vector) of input vector and desired output vector
//! let (x, y) = d1.mean();
//!
//! ```
//!

pub mod activators;
pub mod estimators;
pub mod data;
pub mod io;

extern crate rand;
extern crate serde;
extern crate serde_json;
extern crate bincode;
extern crate csv;

#[macro_use]
extern crate serde_derive;

use std::fmt;
use std::default::Default;

use data::Extractable;

/// Custom ErrorKind enum for handling multiple error types
#[derive(Debug)]
pub enum ErrorKind {
    IO(std::io::Error),
    Encoding(bincode::Error),
    Json(serde_json::Error),
    StdError(Box<dyn std::error::Error>)
}

/// The struct that points different fields of network.
/// It is used only for Display trait. Should be deleted in future versions
#[allow(dead_code)]
enum Field {
    Induced,
    Y,
    Deltas,
    Weights
}

/// This trait should be implemented by neural network structure when you want it
/// to be transformable to other formats. `Note` that you, also, need to implement
/// `serde::Serialize` and `serde::Deserialize` traits before. Hopefully you can
/// do it easily with `derive` attribute.
///
/// Necessity of this trait can be easily described when you restore `FeedForward` instance
/// by `neuroflow::io::load` function. It calls `after` method in order to adjust
/// activation function of neural network.
pub trait Transform: serde::Serialize + for <'de> serde::Deserialize<'de>{
    /// The method that should be called before neural network transformation
    fn before(&mut self){}

    /// The method that should be called after neural network transformation
    fn after(&mut self){}
}

/// Struct `Layer` represents single layer of network.
/// It is private and should not be used directly.
#[derive(Serialize, Deserialize)]
struct Layer {
    v: Vec<f64>,
    y: Vec<f64>,
    delta: Vec<f64>,
    prev_delta: Vec<f64>,
    w: Vec<Vec<f64>>,
}

/// This struct is a container for chosen activation function and its derivative.
/// It is useful when in network's serialization in order to skip function
/// in serialization
struct ActivationContainer{
    func: fn(f64) -> f64,
    der: fn(f64) -> f64
}

/// Feed Forward (multilayer perceptron) neural network that is trained
/// by back propagation algorithm.
/// You can use it for approximation and classification tasks as well.
///
/// # Examples
///
/// In order to create `FeedForward` instance call its constructor `new`.
///
/// The constructor accepts slice as an argument. This slice determines
/// the architecture of network.
/// First element in slice is amount of neurons in input layer
/// and the last one is amount of neurons in output layer.
/// Denote, that vector of input data must have the equal length as input
/// layer of FeedForward neural network (the same is for expected output vector).
///
/// ```rust
/// use neuroflow::FeedForward;
///
/// let mut nn = FeedForward::new(&[1, 3, 2]);
/// ```
///
/// Then you can train your network simultaneously via `fit` method:
///
/// ```rust
/// # use neuroflow::FeedForward;
/// # let mut nn = FeedForward::new(&[1, 3, 2]);
/// nn.fit(&[1.2], &[0.2, 0.8]);
/// ```
///
/// Or to use `train` method with `neuroflow::data::DataSet` struct:
///
/// ```rust
/// # use neuroflow::FeedForward;
/// # let mut nn = FeedForward::new(&[1, 3, 2]);
/// use neuroflow::data::DataSet;
///
/// let mut data = DataSet::new();
/// data.push(&[1.2], &[1.3, -0.2]);
/// nn.train(&data, 30_000); // 30_000 is iterations count
/// ```
///
/// It is possible to set parameters of network:
/// ```rust
/// # use neuroflow::FeedForward;
/// # let mut nn = FeedForward::new(&[1, 3, 2]);
/// nn.learning_rate(0.1)
///   .momentum(0.05)
///   .activation(neuroflow::activators::Type::Tanh);
/// ```
///
/// Call method `calc` in order to calculate value by your(already trained) network:
///
/// ```rust
/// # use neuroflow::FeedForward;
/// # let mut nn = FeedForward::new(&[1, 3, 2]);
/// let d: Vec<f64> = nn.calc(&[1.02]).to_vec();
/// ```
///
#[derive(Serialize, Deserialize)]
pub struct FeedForward {
    layers: Vec<Layer>,
    learn_rate: f64,
    momentum: f64,
    error: f64,

    act_type: activators::Type,

    #[serde(skip_deserializing, skip_serializing)]
    act: ActivationContainer
}

impl Layer {
    fn new(amount: i32, input: i32) -> Layer {
        let mut nl = Layer {v: vec![], y: vec![], delta: vec![], prev_delta: vec![], w: Vec::new()};
        let mut v: Vec<f64>;
        for _ in 0..amount {
            nl.y.push(0.0);
            nl.delta.push(0.0);
            nl.v.push(0.0);

            v = Vec::new();
            for _ in 0..input + 1{
                v.push(2f64 * rand::random::<f64>() - 1f64);
            }

            nl.w.push(v);
        }
        return nl;
    }

    fn bind(&mut self, index: usize){
        self.v.insert(index, 0.0);
        self.y.insert(index, 0.0);
        self.delta.insert(index, 0.0);

        let mut v: Vec<f64> = Vec::new();
        let len = self.w[index].len();

        for _ in 0..len{
            v.push(2f64 * rand::random::<f64>() - 1f64);
        }
        self.w.insert(index, v);
    }

    fn unbind(&mut self, index: usize){
        self.v.remove(index);
        self.y.remove(index);
        self.delta.remove(index);
        self.w.remove(index);
    }
}

impl FeedForward {
    /// The constructor of `FeedForward` struct
    ///
    /// * `architecture: &[i32]` - the architecture of network where each
    /// element in slice represents amount of neurons in this layer.
    /// First element in slice is amount of neurons in input layer
    /// and the last one is amount of neurons in output layer.
    /// Denote, that vector of input data must have the equal length as input
    /// layer of FeedForward neural network (the same is for expected output vector).
    ///
    /// * `return` - `FeedForward` struct
    /// # Example
    ///
    /// ```rust
    /// use neuroflow::FeedForward;
    /// let mut nn = FeedForward::new(&[1, 3, 2]);
    /// ```
    ///
    pub fn new(architecture: &[i32]) -> FeedForward {
        let mut nn = FeedForward {learn_rate: 0.1, momentum: 0.1, error: 0.0,
            layers: Vec::new(),
            act: ActivationContainer{func: activators::tanh, der: activators::der_tanh},
            act_type: activators::Type::Tanh};

        for i in 1..architecture.len() {
            nn.layers.push(Layer::new(architecture[i], architecture[i - 1]))
        }

        return nn;
    }

    fn forward(&mut self, x: &Vec<f64>){
        let mut sum: f64;

        for j in 0..self.layers.len(){
            if j == 0{
                for i in 0..self.layers[j].v.len(){
                    sum = 0.0;
                    for k in 0..x.len(){
                        sum += self.layers[j].w[i][k] * x[k];
                    }
                    self.layers[j].v[i] = sum;
                    self.layers[j].y[i] = (self.act.func)(sum);
                }
            }
            else if j == self.layers.len() - 1{
                for i in 0..self.layers[j].v.len(){
                    sum = self.layers[j].w[i][0];
                    for k in 0..self.layers[j - 1].y.len(){
                        sum += self.layers[j].w[i][k + 1] * self.layers[j - 1].y[k];
                    }
                    self.layers[j].v[i] = sum;
                    self.layers[j].y[i] = sum;
                }
            }
            else {
                for i in 0..self.layers[j].v.len(){
                    sum = self.layers[j].w[i][0];
                    for k in 0..self.layers[j - 1].y.len(){
                        sum += self.layers[j].w[i][k + 1] * self.layers[j - 1].y[k];
                    }
                    self.layers[j].v[i] = sum;
                    self.layers[j].y[i] = (self.act.func)(sum);
                }
            }
        }
    }

    fn backward(&mut self, d: &Vec<f64>){
        let mut sum: f64;

        for j in (0..self.layers.len()).rev(){
            self.layers[j].prev_delta = self.layers[j].delta.clone();
            if j == self.layers.len() - 1{
                self.error = 0.0;
                for i in 0..self.layers[j].y.len(){
                    self.layers[j].delta[i] = (d[i] - self.layers[j].y[i])* (self.act.der)(self.layers[j].v[i]);
                    self.error += 0.5 * (d[i] - self.layers[j].y[i]).powi(2);
                }
            } else {
                for i in 0..self.layers[j].delta.len(){
                    sum = 0.0;
                    for k in 0..self.layers[j + 1].delta.len(){
                        sum += self.layers[j + 1].delta[k] * self.layers[j + 1].w[k][i + 1];
                    }
                    self.layers[j].delta[i] = (self.act.der)(self.layers[j].v[i]) * sum;
                }
            }
        }
    }

    fn update(&mut self, x: &Vec<f64>){
        for j in 0..self.layers.len(){
            for i in 0..self.layers[j].w.len(){
                for k in 0..self.layers[j].w[i].len(){
                    if j == 0 {
                        self.layers[j].w[i][k] += self.learn_rate * self.layers[j].delta[i]*x[k];
                    } else {
                        if k == 0{
                            self.layers[j].w[i][k] += self.learn_rate * self.layers[j].delta[i];
                        } else {
                            self.layers[j].w[i][k] += self.learn_rate * self.layers[j].delta[i]*self.layers[j - 1].y[k - 1];
                        }
                    }
                    self.layers[j].w[i][k] += self.momentum * self.layers[j].prev_delta[i];
                }
            }
        }
    }

    /// Bind a new neuron to layer. It initializes neuron with
    /// random weights.
    ///
    /// * `layer: usize` - index of layer. NOTE, layer indexing starts from 1!
    /// * `neuron: usize` - index of neuron. NOTE, neurons indexing in layer starts from 0!
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use neuroflow::FeedForward;
    /// # let mut nn = FeedForward::new(&[1, 3, 2]);
    /// nn.bind(2, 0);
    /// ```
    pub fn bind(&mut self, layer: usize, neuron: usize){
        self.layers[layer - 1].bind(neuron);
    }

    /// Unbind neuron from layer.
    ///
    /// * `layer: usize` - index of layer. NOTE, layer indexing starts from 1!
    /// * `neuron: usize` - index of neuron. NOTE, neurons indexing in layer starts from 0!
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use neuroflow::FeedForward;
    /// # let mut nn = FeedForward::new(&[1, 3, 2]);
    /// nn.unbind(2, 0);
    /// ```
    pub fn unbind(&mut self, layer: usize, neuron: usize){
        self.layers[layer - 1].unbind(neuron);
    }

    /// Train neural network by bulked data.
    ///
    /// * `data: &T` - the link on data that implements `neuroflow::data::Extractable` trait;
    /// * `iterations: i64` - iterations count.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use neuroflow::FeedForward;
    /// # let mut nn = FeedForward::new(&[1, 3, 2]);
    /// let mut d = neuroflow::data::DataSet::new();
    /// d.push(&[1.2], &[1.3, -0.2]);
    /// nn.train(&d, 30_000);
    /// ```
    pub fn train<T>(&mut self, data: &T, iterations: i64) where T: Extractable{
        for _ in 0..iterations{
            let (x, y) = data.rand();
            self.fit(&x, &y);
        }
    }

    /// Train neural network simultaneously step by step
    ///
    /// * `X: &[f64]` - slice of input data;
    /// * `d: &[f64]` - expected output.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use neuroflow::FeedForward;
    /// # let mut nn = FeedForward::new(&[1, 3, 2]);
    /// nn.fit(&[3.0], &[3.0, 5.0]);
    /// ```
    #[allow(non_snake_case)]
    pub fn fit(&mut self, X: &[f64], d: &[f64]){
        let mut x = X.to_vec();
        let res = d.to_vec();

        x.insert(0, 1f64);

        self.forward(&x);
        self.backward(&res);
        self.update(&x);
    }

    /// Calculate the response by trained neural network.
    ///
    /// * `X: &[f64]` - slice of input data;
    /// * `return -> &[f64]` - slice of calculated data.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use neuroflow::FeedForward;
    /// # let mut nn = FeedForward::new(&[1, 3, 2]);
    /// let v: Vec<f64> = nn.calc(&[1.02]).to_vec();
    /// ```
    #[allow(non_snake_case)]
    pub fn calc(&mut self, X: &[f64]) -> &[f64]{
        let mut x = X.to_vec();

        x.insert(0, 1f64);

        self.forward(&x);
        &self.layers[self.layers.len() - 1].y
    }

    /// Choose activation function. `Note` that if you pass `activators::Type::Custom`
    /// as argument of this method, the default value (`activators::Type::Tanh`) will
    /// be used.
    ///
    /// * `func: neuroflow::activators::Type` - enum element that indicates which
    /// function to use;
    /// * `return -> &mut FeedForward` - link on the current struct.
    pub fn activation(&mut self, func: activators::Type) -> &mut FeedForward{
        match func{
            activators::Type::Sigmoid => {
                self.act_type = activators::Type::Sigmoid;
                self.act.func = activators::sigm;
                self.act.der = activators::der_sigm;
            }
            activators::Type::Tanh | activators::Type::Custom => {
                self.act_type = activators::Type::Tanh;
                self.act.func = activators::tanh;
                self.act.der = activators::der_tanh;
            }
            activators::Type::Relu => {
                self.act_type = activators::Type::Relu;
                self.act.func = activators::relu;
                self.act.der = activators::der_relu;
            }
        }
        self
    }

    /// Set custom activation function and its derivative.
    /// Activation type is set to `activators::Type::Custom`.
    ///
    /// * `func: fn(f64) -> f64` - activation function to be set;
    /// * `der: fn(f64) -> f64` - derivative of activation function;
    /// * `return -> &mut FeedForward` - link on the current struct.
    ///
    /// # Warning
    ///
    /// Be careful using custom activation function. For good results this function
    /// should be smooth, non-decreasing, and differentiable.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use neuroflow::FeedForward;
    ///
    /// fn sigmoid(x: f64) -> f64{
    ///     1.0/(1.0 + x.exp())
    /// }
    ///
    /// fn der_sigmoid(x: f64) -> f64{
    ///     sigmoid(x)*(1.0 - sigmoid(x))
    /// }
    ///
    /// let mut nn = FeedForward::new(&[1, 3, 2]);
    /// nn.custom_activation(sigmoid, der_sigmoid);
    /// ```
    pub fn custom_activation(&mut self, func: fn(f64) -> f64, der: fn(f64) -> f64) -> &mut FeedForward{
        self.act_type = activators::Type::Custom;

        self.act.func = func;
        self.act.der = der;

        self
    }

    /// Set the learning rate of network.
    ///
    /// * `learning_rate: f64` - learning rate;
    /// * `return -> &mut FeedForward` - link on the current struct.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use neuroflow::FeedForward;
    /// # let mut nn = FeedForward::new(&[1, 3, 2]);
    /// nn.learning_rate(0.1);
    /// ```
    pub fn learning_rate(&mut self, learning_rate: f64) -> &mut FeedForward {
        self.learn_rate = learning_rate;
        self
    }

    /// Set the momentum of network.
    ///
    /// * `momentum: f64` - momentum;
    /// * `return -> &mut FeedForward` - link on the current struct.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use neuroflow::FeedForward;
    /// # let mut nn = FeedForward::new(&[1, 3, 2]);
    /// nn.momentum(0.05);
    /// ```
    pub fn momentum(&mut self, momentum: f64) -> &mut FeedForward {
        self.momentum = momentum;
        self
    }

    /// Get current training error
    ///
    /// * `return -> f64` - training error
    pub fn get_error(&self) -> f64{
        self.error
    }
}

impl Transform for FeedForward{
    fn after(&mut self){
        match self.act_type {
            activators::Type::Sigmoid => {
                self.act_type = activators::Type::Sigmoid;
                self.act.func = activators::sigm;
                self.act.der = activators::der_sigm;
            }
            activators::Type::Tanh | activators::Type::Custom => {
                self.act_type = activators::Type::Tanh;
                self.act.func = activators::tanh;
                self.act.der = activators::der_tanh;
            }
            activators::Type::Relu => {
                self.act_type = activators::Type::Relu;
                self.act.func = activators::relu;
                self.act.der = activators::der_relu;
            }
        }
    }
}

impl Default for ActivationContainer{
    fn default() -> ActivationContainer {
        ActivationContainer{func: activators::tanh, der: activators::der_tanh}
    }
}

impl fmt::Display for FeedForward {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result{
        let mut buf: String = format!("**Induced field**\n");

        for v in self.layers.iter(){
            for val in v.v.iter(){
                buf += &format!("{:.3} ", val);
            }
            buf += "\n";
        }
        buf += "\n";

        buf += "**Activated field**\n";
        for v in self.layers.iter(){
            for val in v.y.iter(){
                buf += &format!("{:.3} ", val);
            }
            buf += "\n";
        }
        buf += "\n";

        buf += "**Deltas**\n";
        for v in self.layers.iter(){
            for val in v.delta.iter(){
                buf += &format!("{:.3} ", val);
            }
            buf += "\n";
        }
        buf += "\n";

        buf += "**Weights**\n";
        for v in self.layers.iter() {
            for val in v.w.iter() {
                buf += "[";
                for cell in val.iter() {
                    buf += &format!("{:.3} ", cell);
                }
                buf += "]";
            }
            buf += "\n";
        }

        buf.fmt(f)
    }
}
