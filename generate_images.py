import argparse
import os
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from datetime import datetime
import numpy as np
from PIL import Image
import math

def parse_args():
    parser = argparse.ArgumentParser(description="Generate images using Stable Diffusion")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for image generation")
    parser.add_argument("--negative_prompt", type=str, default=None, 
                        help="Negative prompt to specify what you don't want in the images")
    parser.add_argument("--num_images", type=int, default=8, help="Number of images to generate")
    parser.add_argument("--model", type=str, default="stabilityai/stable-diffusion-2-1", 
                        help="Model to use for image generation")
    parser.add_argument("--output_dir", type=str, default="outputs/stable-diffusion-generations", 
                        help="Directory to save generated images")
    parser.add_argument("--height", type=int, default=768, help="Image height")
    parser.add_argument("--width", type=int, default=768, help="Image width")
    parser.add_argument("--guidance_scale", type=float, default=7.5, 
                        help="Guidance scale for classifier-free guidance")
    parser.add_argument("--num_inference_steps", type=int, default=50, 
                        help="Number of denoising steps")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for generation")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device to CUDA if available, else CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load the pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False
    )
    
    # Use DPMSolver scheduler for faster inference
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    
    # Enable memory efficient attention if xformers is available
    pipe.enable_xformers_memory_efficient_attention()
    
    # Move pipeline to device
    pipe = pipe.to(device)
    
    # Create a timestamp for this generation run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Set base seed if provided
    base_seed = args.seed if args.seed is not None else np.random.randint(0, 2**32)
    
    # Generate images
    images = []
    seeds = []
    
    # Prepare batches based on available GPUs
    num_gpus = torch.cuda.device_count()
    batch_size = min(num_gpus, args.num_images)  # Use as many GPUs as available, up to num_images
    
    if batch_size > 1:
        print(f"Using {batch_size} GPUs for parallel generation")
    
    for i in range(0, args.num_images, batch_size):
        # Generate a batch
        curr_batch_size = min(batch_size, args.num_images - i)
        
        # Generate seeds for this batch
        batch_seeds = [base_seed + j for j in range(i, i + curr_batch_size)]
        seeds.extend(batch_seeds)
        
        # Set the generator for reproducibility
        generators = [torch.Generator(device=device).manual_seed(seed) for seed in batch_seeds]
        
        # Generate images
        batch_images = pipe(
            [args.prompt] * curr_batch_size,
            negative_prompt=[args.negative_prompt] * curr_batch_size if args.negative_prompt else None,
            height=args.height,
            width=args.width,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            generator=generators,
        ).images
        
        images.extend(batch_images)
    
    # Save individual images
    for idx, (image, seed) in enumerate(zip(images, seeds)):
        filename = f"{timestamp}_seed-{seed}_{idx+1}of{args.num_images}.png"
        filepath = os.path.join(args.output_dir, filename)
        image.save(filepath)
        print(f"Saved image {idx+1}/{args.num_images} to {filepath}")
    
    # Create a grid of all generated images
    if args.num_images > 1:
        # Calculate grid dimensions
        grid_size = math.ceil(math.sqrt(args.num_images))
        grid_w, grid_h = grid_size, grid_size
        
        # Create a blank grid image
        grid_image = Image.new('RGB', (grid_w * args.width // 2, grid_h * args.height // 2))
        
        # Place each image in the grid
        for idx, image in enumerate(images):
            if idx >= args.num_images:
                break
                
            # Calculate position in grid
            x = (idx % grid_w) * (args.width // 2)
            y = (idx // grid_w) * (args.height // 2)
            
            # Resize image to fit grid
            resized_image = image.resize((args.width // 2, args.height // 2))
            
            # Place image in grid
            grid_image.paste(resized_image, (x, y))
        
        # Save grid image
        grid_filename = f"{timestamp}_grid_{args.num_images}_images.png"
        grid_filepath = os.path.join(args.output_dir, grid_filename)
        grid_image.save(grid_filepath)
        print(f"Saved grid image to {grid_filepath}")
    
    print(f"\nAll images generated and saved to {args.output_dir}")
    print(f"Prompt: {args.prompt}")
    if args.negative_prompt:
        print(f"Negative prompt: {args.negative_prompt}")

if __name__ == "__main__":
    main()