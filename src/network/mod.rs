
fn act(x: f64) -> f64{
    x.tanh()
}

fn der_act(x: f64) -> f64{
    1.0 - x.tanh().powi(2)
}


pub struct NeuralLayer{
    pub v: Vec<f64>,
    pub y: Vec<f64>,
    pub delta: Vec<f64>,
    pub w: Vec<Vec<f64>>,
}


pub struct NeuralNet{
    pub layers: Vec<NeuralLayer>,
    pub learn_rate: f64,
    pub moment: f64
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
    pub fn new(architecture: Vec<i32>, l_rate: f64, moment: f64) -> NeuralNet {
        let mut nn = NeuralNet{learn_rate: l_rate, moment: moment, layers: Vec::new()};
        for v in architecture.iter() {
            nn.layers.push(NeuralLayer::new(*v))
        }

        return nn;
    }

    pub fn fit(&mut self, X: Vec<f64>, d: Vec<f64>){
        let mut sum: f64;

        // TODO: prepend 1 to X here

        for j in 0..self.layers.len(){
            if j == 0{
                for i in 0..self.layers[j].v.len(){
                    sum = 0.0;
                    for k in 00..X.len(){
                        sum += self.layers[j].w[i][k] * X[k];
                    }
                    self.layers[j].v[i] = sum;
                    self.layers[j].y[i] = act(self.layers[j].v[i]);
                }
            }
                else {
                    for i in 0..self.layers[j].v.len(){
                        sum = 0.0;
                        for k in 00..self.layers[j - 1].y.len(){
                            sum += self.layers[j].w[i][k] * self.layers[j - 1].y[k];
                        }
                        self.layers[j].v[i] = sum;
                        self.layers[j].y[i] = act(self.layers[j].v[i]);
                    }
                }
        }

        for j in self.layers.len() - 1..0 {
            if j == self.layers.len() - 1{
                for i in 0..self.layers[j].y.len(){
                    self.layers[j].y[i] = (self.layers[j].y[i] - d[i])*der_act(self.layers[j].v[i]);
                }
            }
                else {
                    for i in 0..self.layers[j + 1].delta.len(){
                        sum = 0.0;
                        for k in 0..self.layers[j + 1].delta.len(){
                            sum += self.layers[j + 1].delta[i] * self.layers[j + 1].w[k][i + 1];
                        }
                        self.layers[j + 1].delta[i] = der_act(self.layers[j + 1].v[i]) * sum;
                    }
                }
        }

        for j in 0..self.layers.len(){
            for i in 0..self.layers[j].w.len(){
                for k in 0..self.layers[j].w[0].len(){
                    if j == 0 {
                        self.layers[j].w[i][k] += self.learn_rate * self.layers[j].delta[i]*X[k];
                    }
                        else {
                            self.layers[j].w[i][k] += self.learn_rate * self.layers[j].delta[i]*self.layers[j].y[k];
                        }
                }
            }
        }
    }

    pub fn calc(&self, X: Vec<f64>){

    }
}