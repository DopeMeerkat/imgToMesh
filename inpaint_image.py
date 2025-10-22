import argparse
import os
import torch
from diffusers import StableDiffusionInpaintPipeline, DPMSolverMultistepScheduler
from datetime import datetime
import numpy as np
from PIL import Image, ImageOps
import math
from pathlib import Path
import io

# Import rembg, with helpful error messages if it's not installed
try:
    from rembg import remove
except ImportError:
    print("Error: `rembg` package not found.")
    print("Please install it by running: pip install rembg")
    exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description="Inpaint an image using Stable Diffusion")
    parser.add_argument(
        "--input_image", type=str, required=True, help="Path to the input image"
    )
    parser.add_argument(
        "--mask_image", type=str, default=None, help="Optional: Path to the mask image. If not provided, a mask will be auto-generated using rembg."
    )
    parser.add_argument(
        "--prompt", type=str, required=True, help="Text prompt to guide the inpainting"
    )
    parser.add_argument(
        "--negative_prompt", type=str, default=None, help="Negative prompt to specify what to avoid"
    )
    parser.add_argument(
        "--num_variations", type=int, default=4, help="Number of variations to generate"
    )
    parser.add_argument(
        "--model", type=str, default="runwayml/stable-diffusion-v1-5", help="Model to use for inpainting"
    )
    parser.add_argument(
        "--output_dir", type=str, default="outputs/inpainted-images", help="Directory to save inpainted images"
    )
    parser.add_argument(
        "--guidance_scale", type=float, default=7.5, help="Guidance scale"
    )
    parser.add_argument(
        "--num_inference_steps", type=int, default=50, help="Number of denoising steps"
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducibility"
    )
    return parser.parse_args()

def create_grid(images, rows, cols, resize_factor=0.5):
    if not images:
        return None
    
    width, height = images[0].size
    new_width, new_height = int(width * resize_factor), int(height * resize_factor)
    
    grid = Image.new('RGB', size=(cols * new_width, rows * new_height))
    
    for i, img in enumerate(images):
        if i >= rows * cols:
            break
        x = (i % cols) * new_width
        y = (i // cols) * new_height
        resized = img.resize((new_width, new_height), Image.LANCZOS)
        grid.paste(resized, (x, y))
    
    return grid

def main():
    args = parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        input_image = Image.open(args.input_image).convert("RGB")
        print(f"Loaded input image: {args.input_image}")
    except Exception as e:
        print(f"Error loading input image: {e}")
        return

    if args.mask_image:
        # Load user-provided mask
        try:
            mask_image = Image.open(args.mask_image).convert("L")
            print(f"Loaded mask image: {args.mask_image}")
        except Exception as e:
            print(f"Error loading mask image: {e}")
            return
    else:
        # Auto-generate mask using rembg
        print("No mask image provided. Auto-generating mask using rembg...")
        try:
            # Remove background to get an RGBA image
            foreground_image_data = remove(input_image)
            
            # The alpha channel is our mask (foreground is white)
            alpha_mask = foreground_image_data.getchannel('A')

            # Invert the mask: we want to inpaint the BACKGROUND
            mask_image = ImageOps.invert(alpha_mask)
            print("Mask generated and inverted successfully.")
            
            # Save the inverted mask for debugging
            mask_save_path = os.path.join(args.output_dir, f"{Path(args.input_image).stem}_auto_mask_inverted.png")
            mask_image.save(mask_save_path)
            print(f"Saved auto-generated inverted mask to: {mask_save_path}")

        except Exception as e:
            print(f"Error auto-generating mask: {e}")
            return

    if input_image.size != mask_image.size:
        print("Error: Input image and mask image must have the same dimensions.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
    ).to(device)
    
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_xformers_memory_efficient_attention()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_seed = args.seed if args.seed is not None else np.random.randint(0, 2**32)
    
    print(f"Inpainting with prompt: '{args.prompt}'")
    
    inpainted_images = []
    seeds = []
    
    for i in range(args.num_variations):
        current_seed = base_seed + i
        generator = torch.Generator(device=device).manual_seed(current_seed)
        seeds.append(current_seed)
        
        print(f"Generating variation {i+1}/{args.num_variations} with seed {current_seed}...")
        
        image = pipe(
            prompt=args.prompt,
            image=input_image,
            mask_image=mask_image,
            negative_prompt=args.negative_prompt,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            generator=generator,
        ).images[0]
        
        inpainted_images.append(image)

    for idx, (image, seed) in enumerate(zip(inpainted_images, seeds)):
        filename = f"{timestamp}_seed-{seed}_variation{idx+1}.png"
        filepath = os.path.join(args.output_dir, filename)
        image.save(filepath)
        print(f"Saved variation {idx+1}/{args.num_variations} to {filepath}")

    if args.num_variations > 1:
        comparison_images = [input_image] + inpainted_images
        total_images = 1 + args.num_variations
        grid_cols = min(4, total_images)
        grid_rows = (total_images + grid_cols - 1) // grid_cols
        grid = create_grid(comparison_images, grid_rows, grid_cols)
        
        if grid:
            grid_filename = f"{timestamp}_comparison_grid.png"
            grid_filepath = os.path.join(args.output_dir, grid_filename)
            grid.save(grid_filepath)
            print(f"Saved comparison grid to {grid_filepath}")

    print(f"\nInpainting complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()