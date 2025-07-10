from PIL import Image

def resize_with_aspect_ratio(image: Image.Image,
                             target_size: tuple[int, int],
                             fill_value: tuple[int, int, int] = (0, 0, 0)) -> tuple[Image.Image, dict]:
    """
    Resize an image to target resolution while preserving aspect ratio.
    Adds borders if necessary to match exact target dimensions.
    
    Args:
        image: PIL Image to resize
        target_size: Tuple of (width, height) for target resolution
        
    Returns:
        PIL Image resized to exact target dimensions
        dict: Dictionary with original and new dimensions
    """
    target_width, target_height = target_size
    original_width, original_height = image.size
    
    # Calculate scaling factor to fit image within target dimensions
    scale_width = target_width / original_width
    scale_height = target_height / original_height
    scale = min(scale_width, scale_height)
    
    # Calculate new dimensions after scaling
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    
    # Resize the image
    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Create new image with target dimensions and paste resized image centered
    final_image = Image.new('RGB', target_size, (0, 0, 0))  # Black background
    
    # Calculate position to center the resized image
    x_offset = (target_width - new_width) // 2
    y_offset = (target_height - new_height) // 2

    meta = dict(original_size=(new_width, new_height),
                target_size=target_size,
                offsets=(x_offset, y_offset))
    
    final_image.paste(resized_image, (x_offset, y_offset))
    
    return final_image, meta
