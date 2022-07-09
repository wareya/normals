use rand::Rng;
use core::marker::PhantomData;

pub(crate) trait Activation : Default + core::fmt::Debug {
    fn activation(x : f32) -> f32;
    fn derivative(x : f32) -> f32;
}

#[derive(Default, Debug)]
pub(crate) struct Sigmoid {
}
impl Activation for Sigmoid {
    fn activation(mut x : f32) -> f32
    {
        x = x.exp();
        x/(x+1.0)
    }
    fn derivative(x : f32) -> f32
    {
        x * (1.0-x)
    }
}

#[derive(Default, Debug)]
pub(crate) struct ReLU {
}
impl Activation for ReLU {
    fn activation(x : f32) -> f32
    {
        x.max(x*0.01)
    }
    fn derivative(x : f32) -> f32
    {
        if x >= 0.0 { 1.0 } else { 0.01 }
    }
}

#[derive(Default, Debug)]
pub(crate) struct Linear {
}
impl Activation for Linear {
    fn activation(x : f32) -> f32
    {
        x
    }
    fn derivative(_x : f32) -> f32
    {
        1.0
    }
}

pub(crate) trait NeuralLayer : core::fmt::Debug {
    fn new(input_count : usize, output_count : usize) -> Self where Self : Sized;
    fn randomize(&mut self);
    fn get_output_data_mut(&mut self) -> &mut Vec<f32>;
    fn get_output_data(&self) -> &Vec<f32>;
    fn get_output_count(&self) -> usize;
    fn get_input_error(&self) -> &Vec<f32>;
    fn feed_forward(&mut self, input_data : &[f32]);
    fn feed_backward(&mut self, input_data : &[f32], output_error : &[f32]);
    fn apply_gradient(&mut self, learn_rate : f32);
}

#[derive(Default, Debug)]
pub(crate) struct DummyLayer {
    output_count : usize,
    output_data : Vec<f32>,
}

impl NeuralLayer for DummyLayer {
    fn new(_input_count : usize, output_count : usize) -> Self where Self : Sized
    {
        Self {
          output_count,
          output_data : vec!(0.0; output_count+1),
        }
    }
    fn randomize(&mut self) {}
    fn get_output_data_mut(&mut self) -> &mut Vec<f32>
    {
        &mut self.output_data
    }
    fn get_output_data(&self) -> &Vec<f32>
    {
        &self.output_data
    }
    fn get_output_count(&self) -> usize
    {
        self.output_count
    }
    fn get_input_error(&self) -> &Vec<f32>
    {
        panic!("Not allowed to call get_input_error on DummyLayer.")
    }
    fn feed_forward(&mut self, _input_data : &[f32])
    {
        let last = self.output_data.len()-1;
        self.output_data[last] = 1.0;
    }
    fn feed_backward(&mut self, _input_data : &[f32], _output_error : &[f32]) {}
    fn apply_gradient(&mut self, _learn_rate : f32) {}
}

#[derive(Default)]
struct NoDebugVec
{
    a : Vec<f32>,
}
impl core::fmt::Debug for NoDebugVec
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result
    {
        write!(f, "<redacted>")
    }
}


#[derive(Default, Debug)]
pub(crate) struct FullyConnected<T> where T : Activation  {
    input_count : usize,
    output_count : usize,
    matrix : NoDebugVec,
    gradient : NoDebugVec,
    gradient_amount : f32,
    output_data : Vec<f32>,
    input_error : Vec<f32>, // for backpropogation
    phantom : PhantomData<T>,
}

impl<T : Activation> NeuralLayer for FullyConnected<T> {
    fn new(input_count : usize, output_count : usize) -> Self where Self : Sized
    {
        Self {
          input_count,
          output_count,
          matrix : NoDebugVec{a:vec!(0.0; (input_count+1)*output_count)},
          gradient : NoDebugVec{a:vec!(0.0; (input_count+1)*output_count)},
          gradient_amount : 0.0,
          output_data : vec!(0.0; output_count+1),
          input_error : vec!(0.0; input_count+1),
          ..Default::default()
        }
    }
    fn randomize(&mut self)
    {
        let mut rng = rand::thread_rng();
        let h = self.output_count;
        let w = self.input_count+1;
        for y in 0..h
        {
            for x in 0..w
            {
                self.matrix.a[y*w + x] = (rng.gen::<f32>() - 0.5f32)*0.2;
            }
        }
    }
    fn get_output_data_mut(&mut self) -> &mut Vec<f32>
    {
        &mut self.output_data
    }
    fn get_output_data(&self) -> &Vec<f32>
    {
        &self.output_data
    }
    fn get_output_count(&self) -> usize
    {
        self.output_count
    }
    fn get_input_error(&self) -> &Vec<f32>
    {
        &self.input_error
    }
    fn feed_forward(&mut self, input_data : &[f32])
    {
        assert!(input_data.len() >= self.input_count+1);
        
        let h = self.output_count;
        let w = self.input_count+1;
        for y in 0..h
        {
            self.output_data[y] = 0.0;
        }
        for y in 0..h
        {
            for x in 0..w
            {
               self.output_data[y] += input_data[x] * self.matrix.a[y*w + x];
            }
        }
        for y in 0..h
        {
            self.output_data[y] = T::activation(self.output_data[y]);
        }
        self.output_data[h] = 1.0; // bias for next layer
    }
    fn feed_backward(&mut self, input_data : &[f32], output_error : &[f32])
    {
        assert!(input_data.len() >= self.input_count+1);
        assert!(output_error.len() >= self.output_count, "{} less than {}", output_error.len(), self.output_count);
        
        //let normalize = self.input_count as f32;
        
        let h = self.output_count;
        let w = self.input_count+1;
        for x in 0..w
        {
            self.input_error[x] = 0.0;
        }
        for y in 0..h
        {
            // delta_cost = delta of cost / delta of output = output error
            // delta_output = delta of output / delta of weighted input = derivative of activation function of output
            let delta_cost = output_error[y];
            let delta_output = T::derivative(self.output_data[y]);
            for x in 0..w
            {
                // input_data = delta of weighted input / delta of weight = input
                // and via chain rule dc/do * do/dz * dz/dw = dc/dw:
                // delta_weight = delta of cost / delta of weight
                let delta_weight = delta_cost * delta_output * input_data[x];// / normalize;
                self.input_error[x] += delta_weight * self.matrix.a[y*w + x];
                self.gradient.a[y*w + x] += delta_weight;
            }
        }
        self.gradient_amount += 1.0;
    }
    fn apply_gradient(&mut self, learn_rate : f32)
    {
        for i in 0..self.matrix.a.len()
        {
            self.matrix.a[i] -= self.gradient.a[i] * learn_rate / self.gradient_amount;
            self.gradient.a[i] = 0.0;
        }
        self.gradient_amount = 0.0;
    }
}

#[allow(dead_code)]
pub fn multishuffle
<A: std::ops::Index<std::ops::RangeFull, Output = [f32]>,
 B: std::ops::Index<std::ops::RangeFull, Output = [f32]>>
    (a : &mut [A], b : &mut [B])
{
    let mut rng = rand::thread_rng();
    
    assert!(a[..].len() == b[..].len());
    let len = a[..].len();
    
    for i in 0..len-1
    {
        let next = rng.gen_range(i+1..len);
        
        let (a1, a2) = a.split_at_mut(next);
        std::mem::swap(&mut a1[i], &mut a2[0]);
        
        let (b1, b2) = b.split_at_mut(next);
        std::mem::swap(&mut b1[i], &mut b2[0]);
    }
}

#[derive(Default, Debug)]
pub(crate) struct Network {
    #[allow(dead_code)]
    input_count : usize,
    output_count : usize,
    layers : Vec<Box<dyn NeuralLayer>>,
    //inputs : Vec<f32>,
    //outputs : Vec<f32>,
    output_data : Vec<f32>,
    output_error : Vec<f32>,
}

impl Network {
    pub(crate) fn new(input_count : usize, output_count : usize) -> Self
    {
        Self {
            input_count,
            output_count,
            layers : vec!(Box::new(DummyLayer::new(0, input_count))),
            //inputs : vec!(0.0; input_count),
            //outputs : vec!(0.0; output_count),
            output_data : vec!(0.0; output_count),
            output_error : vec!(0.0; output_count),
        }
    }
    pub(crate) fn add_layer<T : NeuralLayer + Sized + 'static>(&mut self, output_count : usize)
    {
        let input_count = self.layers.last().unwrap().get_output_count();
        let mut layer = Box::new(T::new(input_count, output_count));
        layer.randomize();
        self.layers.push(layer);
    }
    pub(crate) fn add_output_layer<T : NeuralLayer + Sized + 'static>(&mut self)
    {
        self.add_layer::<T>(self.output_count)
    }
    pub(crate) fn feed_forward(&mut self, inputs : &[f32]) -> &[f32]
    {
        for (i, val) in inputs.iter().enumerate()
        {
            self.layers[0].get_output_data_mut()[i] = *val;
        }
        for i in 1..self.layers.len()
        {
            let (a, b) = self.layers.split_at_mut(i);
            let prev = &a[i-1];
            let next = &mut b[0];
            next.feed_forward(prev.get_output_data());
        }
        self.layers.last().unwrap().get_output_data()
    }
    pub(crate) fn feed_backward(&mut self)
    {
        for i in (1..self.layers.len()).rev()
        {
            let (a, _b) = self.layers.split_at_mut(i);
            let (b, c) = _b.split_at_mut(1);
            let prev = &a[i-1];
            let next = &mut b[0];
            let output_error = if !c.is_empty() { c[0].get_input_error() } else { &self.output_error };
            next.feed_backward(prev.get_output_data(), output_error);
        }
    }
    pub(crate) fn apply_gradient(&mut self, learn_rate : f32)
    {
        for layer in self.layers.iter_mut()
        {
            layer.apply_gradient(learn_rate);
        }
    }
    pub(crate) fn train_on(&mut self, inputs : &[f32], outputs : &[f32])
    {
        self.feed_forward(&inputs);
        for j in 0..self.output_count
        {
            self.output_error[j] = self.get_output_data()[j] - outputs[j];
        }
        self.feed_backward();
    }
    pub(crate) fn fully_train
    <A: std::ops::Index<std::ops::RangeFull, Output = [f32]>,
     B: std::ops::Index<std::ops::RangeFull, Output = [f32]>>
        (&mut self, input_samples : &mut [A], output_samples : &mut [B], stages : usize, learn_rate : f32)
    {
        assert!(input_samples.len() == output_samples.len());
        
        #[allow(unused_variables)]
        let mut rng = rand::thread_rng();
        
        let num_samples = input_samples.len();
        self.output_error = vec!(0.0; self.output_count);
        
        let mut err_sum = [0.0f32; 3];
        let mut err_count = 0.0f32;
        let mut last_used_sample = 0;
        for i in 0..stages
        {
            /*
            if (i) % num_samples == 0
            {
                multishuffle(input_samples, output_samples);
            }
            */
            //let sample = i % num_samples;
            let sample = rng.gen_range(0..num_samples);
            last_used_sample = sample;
            let in_sample = &input_samples[sample][..];
            let out_sample = &output_samples[sample][..];
            self.train_on(in_sample, out_sample);
            
            //let outputs = self.get_output_data().iter().enumerate().map(|(i, x)| x - out_sample[i]).collect::<Vec<_>>();
            //real_error_sum += outputs.iter().map(|x| x*x).sum::<f32>();
            for j in 0..self.output_error.len()
            {
                err_sum[j] += self.output_error[j].abs();
            }
            err_count += 1.0;
            if i % 128 == 0
            {
                self.apply_gradient(learn_rate);
            }
            if i % 10000 == 0
            {
                println!("finished stage {}: error {} {} {}",
                    i,
                    err_sum[0]/err_count,
                    err_sum[1]/err_count,
                    err_sum[2]/err_count,
                );
            }
            err_sum = [0.0f32; 3];
            err_count = 0.0;
        }
        self.output_data = output_samples[last_used_sample][..].to_vec();
    }
    pub(crate) fn get_output_data(&self) -> &[f32]
    {
        let data = self.layers.last().unwrap().get_output_data();
        let data_len = data.len();
        &data[..data_len-1]
    }
}

#[macro_export]
macro_rules! build_network {
    ($input_count:expr, $output_count:expr
     $(, $a:ident $b:ident $c:expr)*
    ) => ( {
        use $crate::neural::*;
        let mut network = Network::new($input_count, $output_count);
        $(network.add_layer::<$a<$b>>($c);)*
        network.add_output_layer::<FullyConnected<Linear>>();
        network
    } );
}
