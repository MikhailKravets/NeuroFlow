<div align="center">
  <img src="https://raw.githubusercontent.com/MikhailKravets/NeuroFlow/master/logo.png"><br><br>
</div>

[![codecov](https://codecov.io/gh/MikhailKravets/NeuroFlow/branch/master/graph/badge.svg)](https://codecov.io/gh/MikhailKravets/NeuroFlow)
[![crates](https://img.shields.io/crates/v/neuroflow.svg)](https://crates.io/crates/neuroflow)

> NeuroFlow is fast Neural Networks (deep learning) Rust crate.
> It relies on three pillars: speed, reliability, and speed again.

...I would write if this library was going to be the second PyTorch from the Rust world.
However, this repository found its place in the educational area and can be
used by young Rustaceans to enter the world of Neural Networks.

## How to use

Let's try to approximate a very simple function `0.5*sin(e^x) - cos(e^(-x))`.

```rust
extern crate neuroflow;

use neuroflow::FeedForward;
use neuroflow::data::DataSet;
use neuroflow::activators::Type::Tanh;


fn main(){
    /*
        Define a neural network with 1 neuron in input layers. The network contains 4 hidden layers.
        And, such as our function returns a single value, it is reasonable to have 1 neuron in the output layer.
    */
    let mut nn = FeedForward::new(&[1, 7, 8, 8, 7, 1]);
    
    /*
        Define DataSet.
        
        DataSet is the Type that significantly simplifies work with neural networks.
        The majority of its functionality is still under development :(
    */
    let mut data: DataSet = DataSet::new();
    let mut i = -3.0;
    
    // Push the data to DataSet (method push accepts two slices: input data and expected output)
    while i <= 2.5 {
        data.push(&[i], &[0.5*(i.exp().sin()) - (-i.exp()).cos()]);
        i += 0.05;
    }
    
    // Here, we set the necessary parameters and train the neural network by our DataSet with 50 000 iterations
    nn.activation(Tanh)
        .learning_rate(0.01)
        .train(&data, 50_000);

    let mut res;
    
    // Let's check the result
    i = 0.0;
    while i <= 0.3{
        res = nn.calc(&[i])[0];
        println!("for [{:.3}], [{:.3}] -> [{:.3}]", i, 0.5*(i.exp().sin()) - (-i.exp()).cos(), res);
        i += 0.07;
    }
}
```

Expected output
```
for [0.000], [-0.120] -> [-0.119]
for [0.070], [-0.039] -> [-0.037]
for [0.140], [0.048] -> [0.050]
for [0.210], [0.141] -> [0.141]
for [0.280], [0.240] -> [0.236]
```

But we don't want to lose our trained network so easily. So, there is functionality to save and restore
neural networks from files.

```rust

    /*
        In order to save neural network into file call function save from neuroflow::io module.
        
        The first argument is the link to the saving neural network;
        The second argument is the path to the file. 
    */
    neuroflow::io::save(&mut nn, "test.flow").unwrap();
    
    /*
        After we have saved the neural network to the file we can restore it by calling
        of load function from neuroflow::io module.
        
        We must specify the type of new_nn variable.
        The only argument of the load function is the path to a file containing
        the neural network
    */
    let mut new_nn: FeedForward = neuroflow::io::load("test.flow").unwrap();
```

----------------------

Classic XOR problem (with no classic input of data)

Let's create a file named `TerribleTom.csv` at the root of the project. This file should have the following innards:

```
0,0,-,0
0,1,-,1
1,0,-,1
1,1,-,0
```

where `-` is the delimiter that separates the input vector from its desired output vector.

```rust
extern crate neuroflow;

use neuroflow::FeedForward;
use neuroflow::data::DataSet;
use neuroflow::activators::Type::Tanh;


fn main(){
    /*
        Define a neural network with 2 neurons in input layers,
        1 hidden layer (with 2 neurons),
        1 neuron in the output layer
    */
    let mut nn = FeedForward::new(&[2, 2, 1]);
    
    // Here we load data for XOR from the file `TerribleTom.csv`
    let mut data = DataSet::from_csv("TerribleTom.csv");
    
    // Set parameters and train the network
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
```toml
[dependencies]
neuroflow = "0.1.3"
```

Then in project root file
```rust
extern crate neuroflow;
```

## License
MIT License

### Attribution
The origami bird from the logo is made by [Freepik](https://www.freepik.com/)

