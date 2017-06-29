pub mod activations;
pub mod estimations;

extern crate rand;


#[allow(dead_code)]
pub enum Field {
    Induced,
    Y,
    Deltas,
    Weights
}

#[allow(dead_code)]
pub enum Activation{
    Sigmoid,
    Tanh
}


struct Layer {
    v: Vec<f64>,
    y: Vec<f64>,
    delta: Vec<f64>,
    w: Vec<Vec<f64>>,
}


pub struct MLP {
    layers: Vec<Layer>,
    learn_rate: f64,
    moment: f64,

    act: fn(f64) -> f64,
    der_act: fn(f64) -> f64
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
}

impl MLP {
    pub fn new(architecture: Vec<i32>,) -> MLP {
        let mut nn = MLP {learn_rate: 0.1, moment: 0.1,
            layers: Vec::new(),
            act: activations::tanh, der_act: activations::der_tanh};

        for i in 0..architecture.len() {
            if i == 0{
                nn.layers.push(Layer::new(architecture[i], architecture[i]))
            }
                else {
                    nn.layers.push(Layer::new(architecture[i], architecture[i - 1]))
                }
        }

        return nn;
    }

    fn forward(&mut self, x: &Vec<f64>, d: &Vec<f64>){
        let mut sum: f64;

        for j in 0..self.layers.len(){
            if j == 0{
                for i in 0..self.layers[j].v.len(){
                    sum = 0.0;
                    for k in 0..x.len(){
                        sum += self.layers[j].w[i][k] * x[k];
                    }
                    self.layers[j].v[i] = sum;
                    self.layers[j].y[i] = (self.act)(self.layers[j].v[i]);
                }
            } else {
                for i in 0..self.layers[j].v.len(){
                    sum = self.layers[j].w[i][0];
                    for k in 0..self.layers[j - 1].y.len(){
                        sum += self.layers[j].w[i][k + 1] * self.layers[j - 1].y[k];
                    }
                    self.layers[j].v[i] = sum;
                    self.layers[j].y[i] = (self.act)(self.layers[j].v[i]);
                }
            }
        }
    }

    fn backward(&mut self, x: &Vec<f64>, d: &Vec<f64>){
        let mut sum: f64;

        for j in (0..self.layers.len()).rev(){
            if j == self.layers.len() - 1{
                for i in 0..self.layers[j].y.len(){
                    self.layers[j].delta[i] = (d[i] - self.layers[j].y[i])* (self.der_act)(self.layers[j].v[i]);
                }
            } else {
                for i in 0..self.layers[j].delta.len(){
                    sum = 0.0;
                    for k in 0..self.layers[j + 1].delta.len(){
                        sum += self.layers[j + 1].delta[k] * self.layers[j + 1].w[k][i + 1];
                    }
                    self.layers[j].delta[i] = (self.der_act)(self.layers[j].v[i]) * sum;
                }
            }
        }
    }

    fn update(&mut self, x: &Vec<f64>, d: &Vec<f64>){
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

    #[allow(non_snake_case)]
    pub fn fit(&mut self, X: &[f64], d: &[f64]){
        let mut x = X.to_vec();
        let res = d.to_vec();

        x.insert(0, 1f64);

        self.forward(&x, &res);
        self.backward(&x, &res);
        self.update(&x, &res);
    }

    #[allow(non_snake_case)]
    pub fn calc(&mut self, X: &[f64]) -> &[f64]{
        let mut x = X.to_vec();

        x.insert(0, 1f64);

        self.forward(&x, &Vec::new());

        &self.layers[self.layers.len() - 1].y
    }

    pub fn activation(&mut self, func: Activation){
        match func{
            Activation::Sigmoid => {
                self.act = activations::sigm;
                self.der_act = activations::der_sigm;
            }
            Activation::Tanh => {
                self.act = activations::tanh;
                self.der_act = activations::der_tanh;
            }
        }
    }
    pub fn set_activation(&mut self, func: &fn(f64)->f64, derivative: &fn(f64)->f64){
        self.act = *func;
        self.der_act = *derivative;
    }

    pub fn print(&self, e: Field){
        match e {
            Field::Induced => {
                println!("**Induced field**");
                for v in self.layers.iter(){
                    for val in v.v.iter(){
                        print!("{} ", val);
                    }
                    println!();
                }
                println!("----------");
            },
            Field::Y => {
                println!("**Activated field**");
                for v in self.layers.iter(){
                    for val in v.y.iter(){
                        print!("{} ", val);
                    }
                    println!();
                }
                println!("----------");
            },
            Field::Deltas => {
                println!("**Deltas**");
                for v in self.layers.iter(){
                    for val in v.delta.iter(){
                        print!("{} ", val);
                    }
                    println!();
                }
                println!("----------");
            },
            Field::Weights => {
                println!("**Weights field**");
                for v in self.layers.iter(){
                    for val in v.w.iter(){
                        print!("[");
                        for cell in val.iter(){
                            print!("{} ", cell);
                        }
                        print!("]");
                    }
                    println!();
                }
                println!("----------");
            },
        }
    }
}