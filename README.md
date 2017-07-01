# NN-Rust
Neural Networks Rust library. For now, the library contains Multilayer Perceptron that is trained by Backpropagation algorithm.

## Code Example
XOR task with nn-rust library
```rust
    /*
      Create Multilayer Perceptron instance with 
      2 neurons in input layer, 2 neurons in first 
      hidden layer and 1 neuron in output layer
    */
    let mut nn = MLP::new(&[2, 2, 1]);
    /*
      Define training set
    */
    let sc = &[
        (&[0f64, 0f64], &[0f64]),
        (&[1f64, 0f64], &[1f64]),
        (&[0f64, 1f64], &[1f64]),
        (&[1f64, 1f64], &[0f64]),
    ];
    let mut k;
    let rnd_range = Range::new(0, sc.len());
    let prev = time::now_utc();
    
    /* 
      Train the network feeding to it randomly 
      choosen sample from training set 
    */
    for _ in 0..20_000{
        k = rnd_range.ind_sample(&mut rand::thread_rng());
        nn.fit(sc[k].0, sc[k].1);
    }
    
    /*
      Check the result for every sample in training set
    */
    for v in sc{
        println!("Res for: [{}, {}], [{}] -> {:?}", v.0[0], v.0[1], v.1[0], nn.calc(v.0));
    }
```

## Motivation
You may ask me, why are you doing it if there are already a lot of much better solutions (even for Rust)? Actually, it is done in order to fill my educational need. But I'll be glad if you find something useful here.

## Installation
Insert into cargo.toml [dependencies] block next line
```
nn_rust = { git = "https://github.com/MikhailKravets/NN-Rust.git" }
```

## API Reference
Comming soon.

## Tests
All the tests lays in the tests directory (give me a boat!)

## License
MIT License
