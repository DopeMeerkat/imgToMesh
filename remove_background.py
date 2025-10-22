#!/usr/bin/env python3
"""
Background removal script for threestudio
Usage: python remove_background.py input_image.jpg
Output: Saves RGBA PNG with background removed to load/images/
"""

import argparse
import os
import sys
from pathlib import Path

def check_and_import_packages():
    """Check and import required packages with better error handling"""
    try:
        from rembg import remove
        from PIL import Image
        return remove, Image
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Make sure you're in the correct conda environment:")
        print("   conda activate threestudio")
        print("2. Try reinstalling the packages:")
        print("   pip install --upgrade rembg pillow")
        print("3. Or install with conda:")
        print("   conda install -c conda-forge pillow")
        print("   pip install rembg")
        sys.exit(1)


def crop_to_square_centered(img):
    """
    Crop image to square centered around non-transparent content
    
    Args:
        img: PIL Image in RGBA mode
        
    Returns:
        PIL Image: Square cropped image
    """
    # Get the bounding box of non-transparent pixels
    # Convert to numpy for easier processing
    import numpy as np
    from PIL import Image
    
    img_array = np.array(img)
    alpha_channel = img_array[:, :, 3]  # Alpha channel
    
    # Find non-transparent pixels (alpha > threshold)
    non_transparent = alpha_channel > 10  # Small threshold to handle anti-aliasing
    
    if not np.any(non_transparent):
        print("⚠️  Warning: No non-transparent pixels found, using full image")
        # If no content found, just center crop the original
        width, height = img.size
        size = min(width, height)
        left = (width - size) // 2
        top = (height - size) // 2
        return img.crop((left, top, left + size, top + size))
    
    # Get bounding box of content
    rows = np.any(non_transparent, axis=1)
    cols = np.any(non_transparent, axis=0)
    
    top_row = np.argmax(rows)
    bottom_row = len(rows) - np.argmax(rows[::-1]) - 1
    left_col = np.argmax(cols)
    right_col = len(cols) - np.argmax(cols[::-1]) - 1
    
    # Calculate content center and size
    content_center_x = (left_col + right_col) // 2
    content_center_y = (top_row + bottom_row) // 2
    content_width = right_col - left_col + 1
    content_height = bottom_row - top_row + 1
    
    # Determine square size (add padding around content)
    content_size = max(content_width, content_height)
    padding_factor = 1.2  # Add 20% padding around content
    square_size = min(
        int(content_size * padding_factor),
        min(img.size)  # Don't exceed original image dimensions
    )
    
    # Calculate crop coordinates centered on content
    half_size = square_size // 2
    crop_left = max(0, content_center_x - half_size)
    crop_top = max(0, content_center_y - half_size)
    crop_right = min(img.size[0], crop_left + square_size)
    crop_bottom = min(img.size[1], crop_top + square_size)
    
    # Adjust if we hit boundaries
    if crop_right - crop_left < square_size:
        crop_left = max(0, crop_right - square_size)
    if crop_bottom - crop_top < square_size:
        crop_top = max(0, crop_bottom - square_size)
    
    print(f"   Original size: {img.size}")
    print(f"   Content bounds: ({left_col}, {top_row}) to ({right_col}, {bottom_row})")
    print(f"   Crop bounds: ({crop_left}, {crop_top}) to ({crop_right}, {crop_bottom})")
    print(f"   Final size: {crop_right - crop_left}x{crop_bottom - crop_top}")
    
    # Crop the image
    cropped = img.crop((crop_left, crop_top, crop_right, crop_bottom))
    
    # If not perfectly square due to boundary constraints, pad with transparent pixels
    final_size = max(cropped.size)
    if cropped.size[0] != cropped.size[1]:
        # Create a square transparent canvas
        square_img = Image.new('RGBA', (final_size, final_size), (0, 0, 0, 0))
        # Paste the cropped image centered on the canvas
        paste_x = (final_size - cropped.size[0]) // 2
        paste_y = (final_size - cropped.size[1]) // 2
        square_img.paste(cropped, (paste_x, paste_y))
        return square_img
    
    return cropped


def remove_background(input_path, output_dir="load/images"):
    """
    Remove background from input image and save as RGBA PNG
    
    Args:
        input_path (str): Path to input image
        output_dir (str): Directory to save output image
    
    Returns:
        str: Path to output image
    """
    # Import packages (delayed import for better error handling)
    remove, Image = check_and_import_packages()
    
    # Check if input file exists
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output filename
    input_name = Path(input_path).stem
    output_filename = f"{input_name}_rgba.png"
    output_path = os.path.join(output_dir, output_filename)
    
    print(f"Processing: {input_path}")
    print(f"Output will be saved to: {output_path}")
    
    try:
        # Read input image
        with open(input_path, 'rb') as input_file:
            input_data = input_file.read()
        
        # Remove background using rembg
        print("Removing background...")
        output_data = remove(input_data)
        
        # Save as PNG with transparency
        with open(output_path, 'wb') as output_file:
            output_file.write(output_data)
        
        # Open image for processing
        with Image.open(output_path) as img:
            # Convert to RGBA if needed
            if img.mode != 'RGBA':
                print(f"Converting to RGBA mode...")
                img = img.convert('RGBA')
            
            # Crop to square centered around non-transparent content
            print("Cropping to square...")
            img_cropped = crop_to_square_centered(img)
            
            # Check if file already exists and handle replacement
            if os.path.exists(output_path):
                print(f"Replacing existing file: {output_path}")
            
            # Save the final cropped image
            img_cropped.save(output_path)
        
        print(f"✅ Successfully processed! Output saved to: {output_path}")
        print(f"📋 You can now use this in threestudio with:")
        print(f"   data.image_path=./{output_path}")
        
        return output_path
        
    except Exception as e:
        print(f"❌ Error processing image: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Remove background from an image and save as RGBA PNG for use with threestudio",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python remove_background.py image.jpg
  python remove_background.py ~/Desktop/photo.png
  python remove_background.py image.jpg --output custom_output/
        """
    )
    
    parser.add_argument(
        "input_image",
        help="Path to input image file"
    )
    
    parser.add_argument(
        "--output", "-o",
        default="load/images",
        help="Output directory (default: load/images)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        print(f"Input: {args.input_image}")
        print(f"Output directory: {args.output}")
    
    try:
        output_path = remove_background(args.input_image, args.output)
        
        if args.verbose:
            # Show image info
            _, Image = check_and_import_packages()
            with Image.open(output_path) as img:
                print(f"\n📊 Image Info:")
                print(f"   Size: {img.size}")
                print(f"   Mode: {img.mode}")
                print(f"   Format: {img.format}")
                
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()