# Neural Network Rust
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
    
    // Define training set
    let sc = &[
        (&[0f64, 0f64], &[0f64]),
        (&[1f64, 0f64], &[1f64]),
        (&[0f64, 1f64], &[1f64]),
        (&[1f64, 1f64], &[0f64]),
    ];
    let mut k;
    let rnd_range = Range::new(0, sc.len());
    
    // Train the network feeding it examples from training set randomly
    for _ in 0..20_000{
        k = rnd_range.ind_sample(&mut rand::thread_rng());
        nn.fit(sc[k].0, sc[k].1);
    }

    let mut res;
    for v in sc{
        res = nn.calc(v.0)[0];
        println!("for [{:.3}, {:.3}], [{:.3}] -> [{:.3}]",
        v.0[0], v.0[1], v.1[0], res);
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
