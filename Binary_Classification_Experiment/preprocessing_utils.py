
import numpy as np
import cv2
from PIL import Image

def load_image(path, target_size=(224, 224)):
    """Load image, convert to RGB, and resize."""
    img = Image.open(path).convert("RGB")
    if target_size:
        img = img.resize(target_size, Image.BILINEAR)
    return np.array(img)

def apply_hair_removal(img):
    """
    Remove hair using morphological BlackHat transformation and inpainting.
    (DullRazor algorithm adaptation)
    """
    if img is None: return None
    
    # 1. Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # 2. Kernel for morphological operations (size tuned for 224x224)
    # For larger images, kernel should be larger.
    # 9x9 is a reasonable balance for 224px.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    
    # 3. BlackHat (Original - Closing) = finds dark details (clean hairs)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    
    # 4. Thresholding to create hair mask
    # Binary threshold: keep pixels that are significantly darker than surroundings
    _, thresh = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    
    # 5. Inpaint using the mask
    # Telea algorithm (INPAINT_TELEA) or NS
    inpainted = cv2.inpaint(img, thresh, 1, cv2.INPAINT_TELEA)
    
    return inpainted

def apply_color_constancy(img):
    """
    Apply Simple Gray World Color Normalization.
    Scales R, G, B channels so their means equal the global grayscale mean.
    """
    if img is None: return None
    
    # Convert to float for calculation
    img_f = img.astype(float)
    
    # Calculate channel means
    avg_r = np.mean(img_f[:, :, 0])
    avg_g = np.mean(img_f[:, :, 1])
    avg_b = np.mean(img_f[:, :, 2])
    
    # Global gray mean
    avg_gray = (avg_r + avg_g + avg_b) / 3.0
    
    # Scaling factors (add small epsilon to avoid div by zero)
    scale_r = avg_gray / (avg_r + 1e-6)
    scale_g = avg_gray / (avg_g + 1e-6)
    scale_b = avg_gray / (avg_b + 1e-6)
    
    # Apply
    img_f[:, :, 0] *= scale_r
    img_f[:, :, 1] *= scale_g
    img_f[:, :, 2] *= scale_b
    
    # Clip and convert back to uint8
    img_norm = np.clip(img_f, 0, 255).astype(np.uint8)
    
    return img_norm

def generate_mask(img):
    """
    Generate binary lesion mask using Otsu's thresholding.
    Wrapper around existing logic, but using CV2 for speed if available.
    """
    if img is None: return None
    
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Otsu
    # Otsu finds a threshold that minimizes intra-class variance
    # We assume lesion is darker than skin, so mask is values < thresh
    thresh_val, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Binary mask: 1 for lesion (darker), 0 for skin (lighter)
    # The return of threshold is (thresh, binary_img).
    # If binary_img has 0 for lesion, we invest it.
    
    # Actually, cv2.threshold(..., THRESH_BINARY) gives:
    # dst(x,y) = maxval if src(x,y) > thresh else 0
    # Since lesion is DARK pixels, we want src < thresh to be 1 (255).
    # So we use THRESH_BINARY_INV
    
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Optional: Morphological Opening/Closing to clean noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    return mask

def process_single_image(path, target_size=(224, 224), use_hair_removal=True, use_color_norm=True):
    """Pipeline: Load -> Resize -> Hair Removal -> Color Norm -> Mask"""
    
    # 1. Load & Resize
    img = load_image(path, target_size)
    
    # 2. Hair Removal
    if use_hair_removal:
        img = apply_hair_removal(img)
        
    # 3. Color Norm
    if use_color_norm:
        img = apply_color_constancy(img)
        
    # 4. Mask
    mask = generate_mask(img)
    
    return img, mask
