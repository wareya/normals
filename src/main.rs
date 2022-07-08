use image::Pixel;
use image::DynamicImage::*;

use core::marker::PhantomData;

fn px(mut x : i32, mut y : i32, image : & image::Rgba32FImage) -> image::Rgba<f32>
{
    x = x.clamp(0, image.width() as i32-1);
    y = y.clamp(0, image.height() as i32-1);
    image.get_pixel(x as u32, y as u32).to_rgba()
}
fn gray (x : i32, y : i32, image : & image::Rgba32FImage) -> f32
{
    let _gamma = 2.2;
    let pixel = px(x, y, image);
    
    (pixel[0] + pixel[1] + pixel[2]) as f32 / 3.0
}
fn to_normal_trivial(color : &image::Rgba32FImage) -> image::Rgba32FImage
{
    let mut normal = image::DynamicImage::new_rgb32f(color.width(), color.height()).to_rgba32f();
    
    
    let red_kernel = [
        [0.5, 0.0, -0.5],
        [1.0, 0.0, -1.0],
        [0.5, 0.0, -0.5f32],
    ];
    let green_kernel = [
        [-0.5, -1.0, -0.5],
        [ 0.0,  0.0,  0.0],
        [ 0.5,  1.0,  0.5f32],
    ];
    
    /*
    let red_kernel = [
        [1.0, 0.0, -1.0],
    ];
    let green_kernel = [
        [-1.0],
        [ 0.0],
        [ 1.0f32],
    ];
    */
    
    macro_rules! sample
    {
        ($x:expr, $y:expr, $kernel:expr) =>
        { {
            let mut sum = 0.0;
            let height = $kernel.len() as i32;
            for y2 in 0..height
            {
                let width = $kernel[y2 as usize].len() as i32;
                for x2 in 0..width
                {
                    let power = $kernel[y2 as usize][x2 as usize];
                    sum += power * gray($x as i32 + x2 - width/2, $y as i32 + y2 - height/2, color);
                }
            }
            sum.clamp(-1.0, 1.0)
        } };
    }
    
    for y in 0..color.height()
    {
        for x in 0..color.width()
        {
            //let mut r = spherical(sample!(x, y, &red_kernel));
            //let mut g = spherical(sample!(x, y, &green_kernel));
            let mut r = sample!(x, y, &red_kernel);
            let mut g = sample!(x, y, &green_kernel);
            let mut b = (-r*r + -g*g + 1.0).clamp(0.0, 1.0).sqrt();
            
            let len = (r*r + g*g + b*b).sqrt();
            r /= len;
            g /= len;
            b /= len;
            
            r = r/2.0 + 0.5;
            g = g/2.0 + 0.5;
            b = b/2.0 + 0.5;
             
            normal.put_pixel(x, y, image::Rgba([r, g, b, 1.0]));
        }
    }
    normal
}

trait Activation : Default {
    fn activation(x : f32) -> f32;
    fn derivative(x : f32) -> f32;
}

#[derive(Default)]
struct Sigmoid {
}
impl Activation for Sigmoid {
    fn activation(mut x : f32) -> f32
    {
        x = x.exp();
        x/(x+1.0)
    }
    fn derivative(mut x : f32) -> f32
    {
        x * (1.0-x)
    }
}

trait NeuralLayer {
    fn new(input_count : usize, output_count : usize) -> Self where Self : Sized;
    fn feed_forward(&mut self, input_data : &Vec<f32>);
    fn feed_backward(&mut self, output_error : &Vec<f32>);
}

#[derive(Default)]
struct FullyConnected<T> where T : Activation  {
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
          input_error : vec!(0.0; input_count),
          ..Default::default()
        }
    }
    fn feed_forward(&mut self, input_data : &Vec<f32>)
    {
        let h = self.output_count;
        let w = self.input_count+1;
        for y in 0..h
        {
            self.output_data[y] = 0.0;
            for x in 0..w
            {
               self. output_data[y] += input_data[x] * self.matrix[y*w + x]
            }
            self.output_data[y] = T::activation(self.output_data[y]);
        }
        self.output_data[h] = 1.0; // bias for next layer
    }
    fn feed_backward(&mut self, output_error : &Vec<f32>)
    {
        let h = self.output_count;
        let w = self.input_count+1;
        for x in 0..w
        {
            self.input_error[w] = 0.0;
        }
        for y in 0..h
        {
            let error_slope = T::derivative(output_error[y]);
            for x in 0..w
            {
                let error = error_slope * self.matrix[y*w + x];// / ( as f32);
                self.input_error[w] += error;
                self.matrix[y*w + x] -= error;
            }
        }
    }
}

struct Network {
    layers : Vec<Box<dyn NeuralLayer>>,
}

fn main()
{
    let inputs = [
      [0.0, 0.0],
      [1.0, 0.0],
      [0.0, 1.0],
      [1.0, 1.0f32],
    ].to_vec();
    let outputs = [
      [0.0],
      [1.0],
      [1.0],
      [0.0],
    ].to_vec();
    
    
    return;
    let args = std::env::args().collect::<Vec<_>>();
    if args.len() < 4
    {
        println!("usage: ./normals in_color.png in_normal.png out.png");
        return;
    }
    let color = image::open(&args[1]).unwrap();
    let ground_truth = image::open(&args[2]).unwrap();
    let mut normal = image::DynamicImage::new_rgb32f(color.width(), color.height()).to_rgba32f();
    let scales = [1,   2,   4,   8,   16,  32,  64u32];
    let powers = [2.0, 0.7, 0.7, 1.0, 1.0, 1.0, 0.5f32];
    
    let mut octaves = vec!();
    
    let strength = 10.0;
    for (i, scale) in scales.into_iter().enumerate()
    {
        let mut color_downscale = color.resize_exact(color.width()/scale, color.height()/scale, image::imageops::FilterType::CatmullRom).to_rgba32f();
        for c in color_downscale.as_flat_samples_mut().samples.iter_mut()
        {
            *c = c.powf(1.0/2.2);
        }
        let normal_downscale = to_normal_trivial(&color_downscale);
        let normal_upscale = ImageRgba32F(normal_downscale).resize_exact(color.width(), color.height(), image::imageops::FilterType::CatmullRom);
        
        for y in 0..normal.height()
        {
            for x in 0..normal.width()
            {
                let a = px(x as i32, y as i32, &normal);
                let b = match normal_upscale { ImageRgba32F(ref img) => px(x as i32, y as i32, &img), _ => panic!() };
                let mut pixel = a;
                let power = powers[i];
                pixel.0[0] = a.0[0] + (b.0[0]*2.0 - 1.0)*power;
                pixel.0[1] = a.0[1] + (b.0[1]*2.0 - 1.0)*power;
                pixel.0[2] = a.0[2] + (b.0[2]*2.0 - 1.0)*power;
                
                normal.put_pixel(x, y, pixel.to_rgba());
            }
        }
        
        octaves.push(normal_upscale);
    }
    
    for y in 0..normal.height()
    {
        for x in 0..normal.width()
        {
            let mut a = px(x as i32, y as i32, &normal);
            let mut r = a.0[0];
            let mut g = a.0[1];
            let mut b = a.0[2];
            
            r = r*strength;
            g = g*strength;
            
            let len = (r*r + g*g + b*b).sqrt();
            
            r /= len;
            g /= len;
            b /= len;
            
            a.0[0] = r/2.0 + 0.5;
            a.0[1] = g/2.0 + 0.5;
            a.0[2] = b/2.0 + 0.5;
            
            normal.put_pixel(x, y, a.to_rgba());
        }
    }
    let normal = ImageRgba8(ImageRgba32F(normal).to_rgba8());
    
    normal.save(&args[3]).unwrap();
}

/*
    // inputs:
    //   3 (rgb)
    // + 1 (gray)
    // + 7*3 (normals at many scales)
    // + 3 ("best guess" normal)
    // = 28
    // output:
    //   3 (ground truth normal)
    
*/