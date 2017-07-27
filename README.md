# NN-Rust
Neural Networks Rust library. For now, the library contains Multilayer Perceptron that is trained by Backpropagation algorithm.

## Code Example
Simple classification task
```rust
    /*
        Define neural network with 2 neurons in input layers,
        2 hidden layers (where each contains 3 and 4 neurons respectively),
        3 neurons in output layer
    */
    let mut nn = MLP::new(&[2, 3, 4, 3]);
    let mut sample;
    
    /*
        Training set contains 2 vectors - input data and expected output for that input
    */
    let mut training_set: Vec<(Vec<f64>, Vec<f64>)> = Vec::new();
    
    /*
        Define satisfactory amount of elements in training set using Widrow's rule of thumb
    */
    let training_amount = (20f64 * estimators::widrows(&[3, 4, 3], 0.8)) as i32;
    
    /*
        Define 3 classes that are normally distributed.
    */
    let c1 = Normal::new(1f64, 0.5); // mean = 1, variance = 0.5
    let c2 = Normal::new(2f64, 1.0); // mean = 2, variance = 1.0
    let c3 = Normal::new(3f64, 0.35); // mean = 3, variance = 0.35
    
    
    /*
        Simultaneously generate value from each class and add it to the training set
    */
    let mut k = 0;
    for _ in 0..training_amount{
        if k == 0{
            training_set.push((vec![c1.ind_sample(&mut rand::thread_rng()), c1.ind_sample(&mut rand::thread_rng())],
                               vec![1f64, 0f64, 0f64]));
            k += 1;
        }
        else if k == 1 {
            training_set.push((vec![c2.ind_sample(&mut rand::thread_rng()), c2.ind_sample(&mut rand::thread_rng())],
                               vec![0f64, 1f64, 0f64]));
            k += 1;
        }
        else if k == 2 {
            training_set.push((vec![c3.ind_sample(&mut rand::thread_rng()), c3.ind_sample(&mut rand::thread_rng())],
                               vec![0f64, 0f64, 1f64]));
            k += 1;
        }
        else {
            k = 0;
        }
    }

    let rnd_range = Range::new(0, training_set.len());
    
    /*
        Use hyperbolic tangent as activation function
    */
    nn.activation(nn_rust::Activator::Tanh);
    
    /*
        Randomly feed the neural network value from training set
    */
    for _ in 0..50_000{
        k = rnd_range.ind_sample(&mut rand::thread_rng());
        nn.fit(&training_set[k].0, &training_set[k].1);
    }

    {
        sample = [c1.ind_sample(&mut rand::thread_rng()), c1.ind_sample(&mut rand::thread_rng())];
        let res = nn.calc(&sample);
        println!("Res for: [{:?}], [1, 0, 0] -> {:?}", sample, res);
    }

    {
        sample = [c2.ind_sample(&mut rand::thread_rng()), c2.ind_sample(&mut rand::thread_rng())];
        let res = nn.calc(&sample);
        println!("Res for: [{:?}], [0, 1, 0] -> {:?}", sample, res);
    }

    {
        sample = [c3.ind_sample(&mut rand::thread_rng()), c3.ind_sample(&mut rand::thread_rng())];
        let res = nn.calc(&sample);
        println!("Res for: [{:?}], [0, 0, 1] -> {:?}", sample, res);
    }
```
Expected result
```
for: [[1.1630001104941605, 1.1159243765400833]], [1, 0, 0] -> [0.9004036298791347, 0.21423674017110195, -0.010395171772725179]
for: [[2.910223592320417, 2.223002513509626]], [0, 1, 0] -> [0.2995672796100093, 0.6364006240915903, 0.1628364994838787]
for: [[3.858661199225414, 3.492064661955157]], [0, 0, 1] -> [0.07877276613964404, -0.06444202949012201, 0.9761674435051991]
```

## Motivation
The library is done for educational purposes. But I'll be glad if you find something useful here.

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
