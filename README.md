# Neural Network Rust

![Build status](https://travis-ci.org/MikhailKravets/Neural-Network-Rust.svg?branch=master)

Neural Networks Rust library. For now, feed forward neural network is implemented.

### Current goals
- Serialize of Neural Network and write it into file
- Implement Optimal Brain Surgery algorithm
- Work with data in files (``csv``, ``xlsx``, etc)

## Code Example
XOR problem
```rust
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

    nn.activation(activators::Type::Tanh)
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
```
Expected result
```
for [0.000, 0.000], [0.000] -> [0.000]
for [1.000, 0.000], [1.000] -> [1.000]
for [0.000, 1.000], [1.000] -> [1.000]
for [1.000, 1.000], [0.000] -> [0.000]
```

## Motivation
The library is done for educational purposes.

## Installation
Insert into cargo.toml [dependencies] block next line
```
nn_rust = { git = "https://github.com/MikhailKravets/NN-Rust.git" }
```

## License
MIT License
