import argparse
import os
import torch
import numpy as np
from PIL import Image
from datetime import datetime
import math
from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionXLImg2ImgPipeline, DPMSolverMultistepScheduler

def parse_args():
    parser = argparse.ArgumentParser(description="Modify images using Stable Diffusion")
    parser.add_argument("--input_image", type=str, required=True, help="Path to input image")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for image modification")
    parser.add_argument("--negative_prompt", type=str, default=None, 
                        help="Negative prompt to specify what you don't want in the images")
    parser.add_argument("--num_images", type=int, default=4, help="Number of variations to generate")
    parser.add_argument("--model", type=str, default="runwayml/stable-diffusion-v1-5", 
                        help="Model to use for image modification")
    parser.add_argument("--output_dir", type=str, default="outputs/stable-diffusion-generations", 
                        help="Directory to save generated images")
    parser.add_argument("--strength", type=float, default=0.75, 
                        help="Strength of modification (0=original image, 1=complete modification)")
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
    
    # Load and prepare the input image
    try:
        init_image = Image.open(args.input_image).convert("RGB")
        print(f"Loaded input image: {args.input_image}")
    except Exception as e:
        print(f"Error loading input image: {e}")
        return
    
    # Set device to CUDA if available, else CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Print GPU information if available
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Available GPUs: {num_gpus}")
        for i in range(num_gpus):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Check if we're using SDXL
    is_sdxl = "xl" in args.model.lower() or "sd3" in args.model.lower()
    
    # Load the appropriate pipeline
    if is_sdxl:
        pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            args.model,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True
        )
    else:
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            args.model,
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False
        )
    
    # Use DPMSolver scheduler for faster inference
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    
    # Enable memory efficient attention if xformers is available
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except:
        print("Xformers not available, using standard attention")
    
    # Move pipeline to device
    pipe = pipe.to(device)
    
    # Create a timestamp for this generation run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Set base seed if provided
    base_seed = args.seed if args.seed is not None else np.random.randint(0, 2**32)
    
    # Print modification details
    print(f"Modifying image with prompt: '{args.prompt}'")
    print(f"Strength: {args.strength} (0=keep original, 1=completely change)")
    if args.negative_prompt:
        print(f"Negative prompt: '{args.negative_prompt}'")
    
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
        
        # Print progress
        print(f"Generating variations {i+1}-{i+curr_batch_size} of {args.num_images}...")
        
        # Generate images with appropriate parameters based on model type
        if is_sdxl:
            batch_images = pipe(
                prompt=[args.prompt] * curr_batch_size,
                negative_prompt=[args.negative_prompt] * curr_batch_size if args.negative_prompt else None,
                image=[init_image] * curr_batch_size,
                strength=args.strength,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
                generator=generators,
                # Add SDXL specific parameters (important!)
                add_text_embeds=None,  # Will be computed automatically
                add_time_ids=None,     # Will be computed automatically
            ).images
        else:
            batch_images = pipe(
                prompt=[args.prompt] * curr_batch_size,
                negative_prompt=[args.negative_prompt] * curr_batch_size if args.negative_prompt else None,
                image=[init_image] * curr_batch_size,
                strength=args.strength,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
                generator=generators,
            ).images
        
        images.extend(batch_images)
    
    # Save individual images
    for idx, (image, seed) in enumerate(zip(images, seeds)):
        filename = f"{timestamp}_seed-{seed}_{os.path.splitext(os.path.basename(args.input_image))[0]}_var{idx+1}.png"
        filepath = os.path.join(args.output_dir, filename)
        image.save(filepath)
        print(f"Saved variation {idx+1}/{args.num_images} to {filepath}")
    
    # Create a grid of all generated images
    if args.num_images > 1:
        # Calculate grid dimensions
        grid_size = math.ceil(math.sqrt(args.num_images))
        grid_w, grid_h = grid_size, grid_size
        
        # Get original dimensions
        width, height = init_image.size
        
        # Create a blank grid image
        grid_image = Image.new('RGB', (grid_w * width // 2, grid_h * height // 2))
        
        # Place each image in the grid
        for idx, image in enumerate(images):
            if idx >= args.num_images:
                break
                
            # Calculate position in grid
            x = (idx % grid_w) * (width // 2)
            y = (idx // grid_w) * (height // 2)
            
            # Resize image to fit grid
            resized_image = image.resize((width // 2, height // 2))
            
            # Place image in grid
            grid_image.paste(resized_image, (x, y))
        
        # Save grid image
        grid_filename = f"{timestamp}_grid_{os.path.splitext(os.path.basename(args.input_image))[0]}.png"
        grid_filepath = os.path.join(args.output_dir, grid_filename)
        grid_image.save(grid_filepath)
        print(f"Saved grid image to {grid_filepath}")
    
    print(f"\nAll variations generated and saved to {args.output_dir}")

if __name__ == "__main__":
    main()