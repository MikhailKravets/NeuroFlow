


fn act(x: f64) -> f64{
    x.tanh()
}

fn der_act(x: f64) -> f64{
    1.0 - x.tanh().powi(2)
}


struct NeuralLayer{
    v: Vec<f64>,
    y: Vec<f64>,
    delta: Vec<f64>,
    w: Vec<Vec<f64>>,
}


struct NeuralNet{
    layers: Vec<NeuralLayer>,
    learn_rate: f32,
    moment: f32
}

impl NeuralLayer{
    fn new(amount: i32) -> NeuralLayer{
        let mut nl = NeuralLayer{v: vec![], y: vec![], delta: vec![], w: Vec::new()};
        for _ in 0..amount {
            nl.y.push(0.0);
            nl.delta.push(0.0);
            nl.v.push(0.0);

            let mut v: Vec<f64> = vec![];
            for i in 0..amount + 1{
                v = Vec::new();
                v.push(0.01*i as f64);
            }

            nl.w.push(v);
        }
        return nl;
    }
}

impl NeuralNet{
    fn new(architecture: Vec<i32>, l_rate: f32, moment: f32) -> NeuralNet {
        let mut nn = NeuralNet{learn_rate: l_rate, moment: moment, layers: Vec::new()};
        for v in architecture.iter() {
            nn.layers.push(NeuralLayer::new(*v))
        }

        return nn;
    }

    fn fit(&self, X: Vec<f64>, d: Vec<f64>){
        // TODO: prepend 1 to X here

        fn forward(){
            for j in 0..self::layers.len(){
                let mut sum;

                if j == 0{
                    for i in 0..self::layers[j].v.len(){
                        sum = 0.0;
                        for k in 00..X.len(){
                            sum += self.layers[j].w[i][k] * X[k];
                        }
                        self.layers[j].v[i] = sum;
                        self.layers[j].y[i] = act(self.layers[j].v[i]);
                    }
                }
                else {
                    for i in 0..self::layers[j].v.len(){
                        sum = 0.0;
                        for k in 00..self.layers[j - 1].y.len(){
                            sum += self.layers[j].w[i][k] * self.layers[j - 1].y[k];
                        }
                        self.layers[j].v[i] = sum;
                        self.layers[j].y[i] = act(self.layers[j].v[i]);
                    }
                }
            }
        }

        fn backward(){

        }

        fn update(){

        }

        forward();
        backward();
        update();
    }

    fn calc(&self, X: Vec<f64>){

    }
}


fn main() {
    println!("Application starts!");

    let nn = NeuralNet::new(vec![2, 2, 1], 0.1, 0.1);

    for v in nn.layers{
        for val in v.y{
            println!("val: {}", val)
        }
        println!("----------")
    }

    println!("Application stops!");
}
