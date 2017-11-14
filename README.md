<div align="center">
  <img src="https://raw.githubusercontent.com/MikhailKravets/DataFlow/master/logo.png"><br><br>
</div>

![Build status](https://travis-ci.org/MikhailKravets/NeuroFlow.svg?branch=master)
[![codecov](https://codecov.io/gh/MikhailKravets/NeuroFlow/branch/master/graph/badge.svg)](https://codecov.io/gh/MikhailKravets/NeuroFlow)
[![crates](https://img.shields.io/crates/v/neuroflow.svg)](https://crates.io/crates/neuroflow)

Neural Networks Rust crate that is based on speed, safety, and clear sanity.

## How to use

Let's try to approximate simple `sin(x)` function.

```rust
extern crate neuroflow;

use neuroflow::FeedForward;
use neuroflow::data::DataSet;
use neuroflow::activators::Type::Tanh;


fn main(){
    /*
        Define neural network with 1 neuron in input layers 
        (we have only 1 argument in sin(x), so it should be 1 neuron in the input layer).
        Network contains 2 hidden layers (that have 8 and 6 neurons respectively).
        And, such as sin(x) returns single value, it is reasonable to have 1 neuron in the output layer.
    */
    let mut nn = FeedForward::new(&[1, 8, 6, 1]);
    
    /*
        Define DataSet.
        
        DataSet is the Type that significantly simplifies work with neural network.
        Majority of its functionality is still under development :(
    */
    let mut data: DataSet = DataSet::new();
    let mut i = -3.0;
    
    // Push the data to DataSet (method push accepts two slices: input data and expected output)
    while i <= 3.0 {
        data.push(&[i], &[i.sin()]);
        i += 0.1;
    }
    
    // Here, we set necessary parameters and train neural network by our DataSet with 30 000 iterations
    nn.activation(Tanh)
        .learning_rate(0.05)
        .train(&data, 30_000);

    let mut res;
    
    // Let's check the result
    i = 0.0;
    while i <= 0.3{
        res = nn.calc(&[i])[0];
        println!("for [{:.3}], [{:.3}] -> [{:.3}]", i, i.sin(), res);
        i += 0.05;
    }
}
```

Expected output
```
for [0.000], [0.000] -> [0.003]
for [0.050], [0.050] -> [0.048]
for [0.100], [0.100] -> [0.098]
for [0.150], [0.149] -> [0.149]
for [0.200], [0.199] -> [0.199]
for [0.250], [0.247] -> [0.248]
for [0.300], [0.296] -> [0.297]
```

But we don't want to lose our trained network so easily. So, there is functionality to save and restore
neural networks from files.

```rust

    /*
        In order to save neural network into file call function save from neuroflow::io module.
        
        First argument is link on the saving neural network;
        Second argument is path to the file. 
    */
    neuroflow::io::save(&nn, "test.flow").unwrap();
    
    /*
        After we have saved the neural network to the file we can restore it by calling
        of load function from neuroflow::io module.
        
        We must specify the type of new_nn variable.
        The only argument of load function is the path to file containing
        the neural network
    */
    let mut new_nn: FeedForward = neuroflow::io::load("test.flow").unwrap();
```

----------------------

Classic XOR problem
```rust
extern crate neuroflow;

use neuroflow::FeedForward;
use neuroflow::data::DataSet;
use neuroflow::activators::Type::Tanh;


fn main(){
    /*
        Define neural network with 2 neurons in input layers,
        1 hidden layer (with 2 neurons),
        1 neuron in output layer
    */
    let mut nn = FeedForward::new(&[2, 2, 1]);
    let mut data = DataSet::new();

    data.push(&[0f64, 0f64], &[0f64]);
    data.push(&[1f64, 0f64], &[1f64]);
    data.push(&[0f64, 1f64], &[1f64]);
    data.push(&[1f64, 1f64], &[0f64]);

    nn.activation(Tanh)
        .learning_rate(0.1)
        .momentum(0.15)
        .train(&data, 20_000);

    let mut res;
    let mut d;
    for i in 0..data.len(){
        res = nn.calc(data.get(i).0)[0];
        d = data.get(i);
        println!("for [{:.3}, {:.3}], [{:.3}] -> [{:.3}]", d.0[0], d.0[1], d.1[0], res);
    }
}
```
Expected output
```
for [0.000, 0.000], [0.000] -> [0.000]
for [1.000, 0.000], [1.000] -> [1.000]
for [0.000, 1.000], [1.000] -> [1.000]
for [1.000, 1.000], [0.000] -> [0.000]
```

## Installation
Insert into your project's cargo.toml block next line
```
[dependencies]
neuroflow = "0.1.2"
```

Then in your code
```rust
extern crate neuroflow;
```

## Motivation
Previously the library was created only for educational purposes. Saying about now there is, also, sport interest :)

## License
MIT License

### Attribution
The origami bird from logo is made by [Freepik](https://www.freepik.com/)

