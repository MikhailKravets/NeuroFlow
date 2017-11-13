pub mod activators;
pub mod estimators;
pub mod data;
pub mod io;

extern crate rand;
extern crate serde;
extern crate bincode;

#[macro_use]
extern crate serde_derive;

use std::fmt;
use std::default::Default;

use data::Extractable;

/// The struct that points different fields of network.
/// It is used only for Display trait. Should be deleted in future versions
#[allow(dead_code)]
enum Field {
    Induced,
    Y,
    Deltas,
    Weights
}

/// Struct `Layer` represents single layer of network.
/// It is private and should not be used directly.
#[derive(Serialize, Deserialize)]
struct Layer {
    v: Vec<f64>,
    y: Vec<f64>,
    delta: Vec<f64>,
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
/// ```rust, no_run
/// let mut nn = FeedForward::new(&[1, 3, 2]);
/// ```
///
/// Then you can train your network simultaneously via `fit` method:
/// ```rust, no_run
/// nn.fit(&[1.2], &[0.2, 0.8]);
/// ```
/// Or to use `train` method with `neuroflow::data::DataSet` struct:
/// ```rust, no_run
/// let mut data = DataSet::new();
/// nn.train(data, 30_000); // 30_000 is iterations count
/// ```
///
/// It is possible to set parameters of network:
/// ```rust, no_run
/// nn.learning_rate(0.1)
///   .momentum(0.05)
///   .activation(neuroflow::activators::Type::Tanh);
/// ```
///
/// Call method `calc` in order to calculate value by your(already trained) network:
/// ```rust, no_run
/// let d: Vec<f64> = nn.calc(&[1.02]).to_vec();
/// ```
///
#[derive(Serialize, Deserialize)]
pub struct FeedForward {
    layers: Vec<Layer>,
    learn_rate: f64,
    momentum: f64,

    act_type: activators::Type,

    #[serde(skip_deserializing, skip_serializing)]
    act: ActivationContainer
}

impl Layer {
    fn new(amount: i32, input: i32) -> Layer {
        let mut nl = Layer {v: vec![], y: vec![], delta: vec![], w: Vec::new()};
        let mut v: Vec<f64>;
        for _ in 0..amount {
            nl.y.push(0.0);
            nl.delta.push(0.0);
            nl.v.push(0.0);

            v = Vec::new();
            for _ in 0..input + 1{
                v.push(2f64*rand::random::<f64>() - 1f64);
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
            v.push(2f64*rand::random::<f64>() - 1f64);
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
    /// ```rust, no_run
    /// let mut nn = FeedForward::new(&[1, 3, 2]);
    /// ```
    ///
    pub fn new(architecture: &[i32]) -> FeedForward {
        let mut nn = FeedForward {learn_rate: 0.1, momentum: 0.1,
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
            if j == self.layers.len() - 1{
                for i in 0..self.layers[j].y.len(){
                    self.layers[j].delta[i] = (d[i] - self.layers[j].y[i])* (self.act.der)(self.layers[j].v[i]);
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
                    }
                        else {
                            if k == 0{
                                self.layers[j].w[i][k] += self.learn_rate * self.layers[j].delta[i];
                            } else {
                                self.layers[j].w[i][k] += self.learn_rate * self.layers[j].delta[i]*self.layers[j - 1].y[k - 1];
                            }
                        }
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
    /// # Example
    /// ```rust, no_run
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
    /// # Example
    /// ```rust, no_run
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
    /// # Example
    /// ```rust, no_run
    /// let mut d = neuroflow::data::DataSet::new();
    /// nn.train(d, 30_000);
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
    /// # Example
    /// ```rust, no_run
    /// nn.fit(&[3], &[3, 5]);
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
    /// # Example
    /// ```rust, no_run
    /// let v: Vec<f64> = nn.calc(&[1.02]).to_vec();
    /// ```
    #[allow(non_snake_case)]
    pub fn calc(&mut self, X: &[f64]) -> &[f64]{
        let mut x = X.to_vec();

        x.insert(0, 1f64);

        self.forward(&x);
        &self.layers[self.layers.len() - 1].y
    }

    /// Choose activation function.
    ///
    /// * `func: neuroflow::activators::Type` - enum element that indicates which
    /// function to use;
    /// * `return -> &mut FeedForward` - link on the current struct.
    pub fn activation(&mut self, func: activators::Type) -> &mut FeedForward{
        match func{
            activators::Type::Sigmoid => {
                self.act.func = activators::sigm;
                self.act.der = activators::der_sigm;
            }
            activators::Type::Tanh => {
                self.act.func = activators::tanh;
                self.act.der = activators::der_tanh;
            }
        }
        self
    }

    /// Set the learning rate of network.
    ///
    /// * `learning_rate: f64` - learning rate;
    /// * `return -> &mut FeedForward` - link on the current struct.
    ///
    /// # Example
    /// ```rust, no_run
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
    /// ```rust, no_run
    /// nn.momentum(0.05);
    /// ```
    pub fn momentum(&mut self, momentum: f64) -> &mut FeedForward {
        self.momentum = momentum;
        self
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