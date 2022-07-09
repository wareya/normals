use image::Pixel;
use image::DynamicImage::*;

use rand::Rng;

use core::marker::PhantomData;

mod neural;

fn px(mut x : i32, mut y : i32, image : & image::Rgba32FImage) -> image::Rgba<f32>
{
    x = x.clamp(0, image.width() as i32-1);
    y = y.clamp(0, image.height() as i32-1);
    image.get_pixel(x as u32, y as u32).to_rgba()
}
fn gray (x : i32, y : i32, image : & image::Rgba32FImage) -> f32
{
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

fn set_up_images(color_fname : &str, ground_truth_fname : &str) -> (image::Rgba32FImage, image::Rgba32FImage, image::Rgba32FImage, Vec<image::Rgba32FImage>)
{
    
    let color = image::open(color_fname).unwrap();
    let ground_truth = if ground_truth_fname != "" { image::open(ground_truth_fname).unwrap().to_rgba32f() } else { image::DynamicImage::new_rgb32f(1, 1).to_rgba32f() };
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
        
        octaves.push(normal_upscale.to_rgba32f());
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
    
    let color = color.to_rgba32f();
    
    (color, ground_truth, normal, octaves)
}

fn set_up_inputs<const N : usize, const M : usize>
    (color_fname : &str, ground_truth_fname : &str, input_samples : &mut Vec<[f32; N]>, output_samples : &mut Vec<[f32; M]>)
{
    let (color, ground_truth, normal, octaves) = set_up_images(color_fname, ground_truth_fname);
    
    let mut rng = rand::thread_rng();
    /*
    for i in 0..50000
    {
        let mut inputs = [0.0f32; N];
        let mut z = 0;
        let mut push_value = |x|
        {
            inputs[z] = x;
            z += 1;
        };
        let y = rng.gen_range(0..normal.height() as i32);
        let x = rng.gen_range(0..normal.width() as i32);
        */
        
    for y in 0..normal.height() as i32
    {
        for x in 0..normal.height() as i32
        {
            let mut inputs = [0.0f32; N];
            let mut z = 0;
            let mut push_value = |x|
            {
                inputs[z] = x - 0.5;
                z += 1;
            };
            
            push_value(gray(x, y, &color));
            
            let mut push_pixel = |x : image::Rgba<f32>|
            {
                inputs[z+0] = x[0] - 0.5;
                inputs[z+1] = x[1] - 0.5;
                inputs[z+2] = x[2] - 0.5;
                z += 3;
            };
            
            push_pixel(px(x, y, &color));
            for octave in &octaves
            {
                push_pixel(px(x, y, &octave));
            }
            push_pixel(px(x, y, &normal));
            
            let output_px = px(x, y, &ground_truth);
            let mut outputs = [0.0f32; M];
            outputs[0] = output_px[0] - 0.5;
            outputs[1] = output_px[1] - 0.5;
            outputs[2] = output_px[2] - 0.5;
            
            input_samples.push(inputs);
            output_samples.push(outputs);
        }
    }
}

fn main()
{
    let args = std::env::args().collect::<Vec<_>>();
    if args.len() < 3
    {
        println!("usage: ./normals in_color.png out.png");
        return;
    }
    
    let mut input_samples  = Vec::<[f32; 28]>::with_capacity(10000);
    let mut output_samples = Vec::<[f32; 3]>::with_capacity(10000);
    let mut network = build_network!(28, 3, FullyConnected Sigmoid 32);
    
    for pair in std::fs::read_to_string("training_pairs.txt").unwrap().lines().map(|x| x.split("\t"))
    {
        let pair = pair.collect::<Vec<_>>();
        set_up_inputs(pair[0], pair[1], &mut input_samples, &mut output_samples);
    }
    
    println!("beginning training");
    //network.fully_train(&mut input_samples, &mut output_samples, 50000, 0.05);
    
    let (color, _, normal, octaves) = set_up_images(&args[1], "");
    
    for _i in 0..2
    {
        let mut normal = image::DynamicImage::new_rgb32f(color.width(), color.height()).to_rgba32f();
        
        for y in 0..normal.height() as i32
        {
            for x in 0..normal.width() as i32
            {
                let mut inputs = [0.0f32; 28];
                let mut z = 0;
                let mut push_value = |x|
                {
                    inputs[z] = x - 0.5;
                    z += 1;
                };
                
                push_value(gray(x, y, &color));
                
                let mut push_pixel = |x : image::Rgba<f32>|
                {
                    inputs[z+0] = x[0] - 0.5;
                    inputs[z+1] = x[1] - 0.5;
                    inputs[z+2] = x[2] - 0.5;
                    z += 3;
                };
                
                push_pixel(px(x, y, &color));
                for octave in &octaves
                {
                    push_pixel(px(x, y, &octave));
                }
                push_pixel(px(x, y, &normal));
                
                let mut outputs = [1.0f32; 4];
                let outputty = network.feed_forward(&inputs);
                outputs[0] = outputty[0] + 0.5;
                outputs[1] = outputty[1] + 0.5;
                outputs[2] = outputty[2] + 0.5;
                
                normal.put_pixel(x as u32, y as u32, image::Rgba::<f32>::from(outputs));
            }
        }
        println!("{:?}", network);
        let normal = ImageRgba8(ImageRgba32F(normal).to_rgba8());
        normal.save(format!("{}_{}", _i, &args[2])).unwrap();
        network.fully_train(&mut input_samples, &mut output_samples, 1, 0.5);
    }
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