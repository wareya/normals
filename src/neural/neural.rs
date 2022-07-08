mod neural;

use rand::Rng;
use core::marker::PhantomData;

pub(crate) trait Activation : Default {
    fn activation(x : f32) -> f32;
    fn derivative(x : f32) -> f32;
}

#[derive(Default)]
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

#[derive(Default)]
pub(crate) struct Linear {
}
impl Activation for Linear {
    fn activation(mut x : f32) -> f32
    {
        x
    }
    fn derivative(x : f32) -> f32
    {
        1.0
    }
}

pub(crate) trait NeuralLayer {
    fn new(input_count : usize, output_count : usize) -> Self where Self : Sized;
    fn randomize(&mut self);
    fn get_output_data_mut(&mut self) -> &mut Vec<f32>;
    fn get_output_data(&self) -> &Vec<f32>;
    fn get_output_count(&self) -> usize;
    fn get_input_error(&self) -> &Vec<f32>;
    fn feed_forward(&mut self, input_data : &[f32]);
    fn feed_backward(&mut self, input_data : &[f32], output_error : &[f32], learn_rate : f32);
}

#[derive(Default)]
pub(crate) struct DummyLayer {
    output_count : usize,
    output_data : Vec<f32>,
}

impl NeuralLayer for DummyLayer {
    fn new(input_count : usize, output_count : usize) -> Self where Self : Sized
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
    fn feed_forward(&mut self, input_data : &[f32])
    {
        let last = self.output_data.len()-1;
        self.output_data[last] = 1.0;
    }
    fn feed_backward(&mut self, input_data : &[f32], output_error : &[f32], learn_rate : f32) {}
}


#[derive(Default)]
pub(crate) struct FullyConnected<T> where T : Activation  {
    input_count : usize,
    output_count : usize,
    matrix : Vec<f32>,
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
          matrix : vec!(0.0; (input_count+1)*output_count),
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
                self.matrix[y*w + x] = rng.gen();
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
               self.output_data[y] += input_data[x] * self.matrix[y*w + x];
            }
        }
        for y in 0..h
        {
            self.output_data[y] = T::activation(self.output_data[y]);
        }
        self.output_data[h] = 1.0; // bias for next layer
    }
    fn feed_backward(&mut self, input_data : &[f32], output_error : &[f32], learn_rate : f32)
    {
        assert!(input_data.len() >= self.input_count+1);
        assert!(output_error.len() >= self.output_count, "{} less than {}", output_error.len(), self.output_count);
        
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
                let delta_weight = delta_cost * delta_output * input_data[x];
                self.input_error[x] += delta_weight * self.matrix[y*w + x];
                self.matrix[y*w + x] -= delta_weight * learn_rate;
            }
        }
    }
}

#[derive(Default)]
pub(crate) struct Network {
    input_count : usize,
    output_count : usize,
    layers : Vec<Box<dyn NeuralLayer>>,
    //inputs : Vec<f32>,
    //outputs : Vec<f32>,
    output_error : Vec<f32>,
}

impl Network {
    fn new(input_count : usize, output_count : usize) -> Self
    {
        Self {
            input_count,
            output_count,
            layers : vec!(Box::new(DummyLayer::new(0, input_count))),
            //inputs : vec!(0.0; input_count),
            //outputs : vec!(0.0; output_count),
            output_error : vec!(0.0; input_count),
        }
    }
    fn add_layer<T : NeuralLayer + Sized + 'static>(&mut self, output_count : usize)
    {
        let input_count = self.layers.last().unwrap().get_output_count();
        let mut layer = Box::new(T::new(input_count, output_count));
        layer.randomize();
        self.layers.push(layer);
    }
    fn add_output_layer<T : NeuralLayer + Sized + 'static>(&mut self)
    {
        self.add_layer::<T>(self.output_count)
    }
    fn feed_forward(&mut self, inputs : &[f32]) -> &[f32]
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
    fn feed_backward(&mut self, learn_rate : f32)
    {
        for i in (1..self.layers.len()).rev()
        {
            let (a, _b) = self.layers.split_at_mut(i);
            let (b, c) = _b.split_at_mut(1);
            let prev = &a[i-1];
            let next = &mut b[0];
            let output_error = if !c.is_empty() { c[0].get_input_error() } else { &self.output_error };
            next.feed_backward(prev.get_output_data(), output_error, learn_rate);
        }
    }
    fn train_on(&mut self, inputs : &[f32], outputs : &[f32], learn_rate : f32)
    {
        self.feed_forward(&inputs);
        for j in 0..self.output_count
        {
            self.output_error[j] = self.get_output_data()[j] - outputs[j];
        }
        self.feed_backward(learn_rate);
    }
    fn fully_train
    <A: std::ops::Index<std::ops::RangeFull, Output = [f32]>,
     B: std::ops::Index<std::ops::RangeFull, Output = [f32]>>
        (&mut self, input_samples : &[A], output_samples : &[B], stages : usize, learn_rate : f32)
    {
        assert!(input_samples.len() == output_samples.len());
        let num_samples = input_samples.len();
        let mut output_error = vec!(0.0; output_samples[0][..].len());
        for i in 0..stages
        {
            let sample = i % num_samples;
            self.train_on(&input_samples[sample][..], &output_samples[sample][..], learn_rate);
        }
    }
    fn get_output_data(&self) -> &[f32]
    {
        let data = self.layers.last().unwrap().get_output_data();
        let data_len = data.len();
        &data[..data_len-1]
    }
}

#[macro_export]
macro_rules! build_network {
    ($input_count:expr, $output_count:expr,
     $($a:ident $b:ident $c:expr $(,)?)*
    ) => ( {
        let mut network = Network::new($input_count, $output_count);
        $(network.add_layer::<$a<$b>>($c);)*
        network.add_output_layer::<FullyConnected<Linear>>();
        network
    } );
}
