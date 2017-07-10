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
    for i in 0..training_amount{
        if k == 0{
            training_set.push((vec![c1.ind_sample(&mut rand::thread_rng()), c1.ind_sample(&mut rand::thread_rng())], vec![1f64, 0f64, 0f64]));
            k += 1;
        }
        else if k == 1 {
            training_set.push((vec![c2.ind_sample(&mut rand::thread_rng()), c2.ind_sample(&mut rand::thread_rng())], vec![0f64, 1f64, 0f64]));
            k += 1;
        }
        else if k == 2 {
            training_set.push((vec![c3.ind_sample(&mut rand::thread_rng()), c3.ind_sample(&mut rand::thread_rng())], vec![0f64, 0f64, 1f64]));
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
Res for: [[1.2154260379961084, 0.5240532938133206]], [1, 0, 0] -> [0.9043892795089641, 0.044860317736932304, 0.01336331265053569]
Res for: [[3.1533061574154067, 1.2676336140495372]], [0, 1, 0] -> [0.16666039429457316, 0.8046737684912082, 0.020950389723868302]
Res for: [[2.9893548907003327, 3.336032369101008]], [0, 0, 1] -> [0.02594222276553783, -0.049839163422333486, 0.9953174852324813]
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
