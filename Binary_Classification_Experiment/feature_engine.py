# feature_engine.py
"""
Consolidated Feature Engine for ExSTraCS Deployment.
Combines:
1. ABCD (Asymmetry, Border, Color, Diameter)
2. Wavelets (Texture)
3. Hu Moments (Shape)
4. Color Auto-Correlograms (Spatial Color)
5. C-HOG (Color-Channel Histogram of Oriented Gradients)

Ensures exact feature order matching 'selected_features_top50.json'.
"""

import numpy as np
import pandas as pd
from PIL import Image
import json
from pathlib import Path

# Scipy is needed for skew/kurtosis if not implementing from scratch
from scipy.stats import skew, kurtosis
import pywt # For Wavelets

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
IMAGE_SIZE = 224

# ------------------------------------------------------------------
# 1. BASIC UTILS
# ------------------------------------------------------------------
def load_image(path_or_file):
    """Load image from path or file-like object (Streamlit upload)"""
    img = Image.open(path_or_file).convert("RGB")
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    return np.array(img, dtype=np.uint8)

def otsu_threshold(gray_img):
    """Otsu's method from scratch."""
    pixels = gray_img.ravel()
    hist, bin_edges = np.histogram(pixels, bins=256, range=(0, 256))
    
    total = pixels.size
    current_max, threshold = 0, 0
    sum_total = np.sum(np.arange(256) * hist)
    sum_b, weight_b = 0, 0
    
    for i in range(256):
        weight_b += hist[i]
        if weight_b == 0: continue
        weight_f = total - weight_b
        if weight_f == 0: break
            
        sum_b += i * hist[i]
        m_b = sum_b / weight_b
        m_f = (sum_total - sum_b) / weight_f
        
        var_between = weight_b * weight_f * (m_b - m_f) ** 2
        if var_between > current_max:
            current_max = var_between
            threshold = i
    return threshold

def get_mask(rgb_img):
    gray = np.dot(rgb_img[...,:3], [0.299, 0.587, 0.114])
    thresh = otsu_threshold(gray)
    mask = gray < thresh
    return mask

# ------------------------------------------------------------------
# 2. FEATURE GROUP: ABCD & COLOR STATS
# ------------------------------------------------------------------
try:
    from scipy.stats import skew, kurtosis
except ImportError:
    # Minimal fallback if scipy missing
    def skew(x): return 0
    def kurtosis(x): return 0

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import pywt
except ImportError:
    pywt = None # Will fail later if wavelets needed

# ------------------------------------------------------------------
# 2. FEATURE GROUP: ABCD & COLOR STATS
# ------------------------------------------------------------------
def extract_abcd_color(rgb, mask):
    feats = {}
    h, w = mask.shape
    
    # --- Asymmetry (Mirror Mismatch Logic from original extract_abcd_features.py) ---
    left_half = mask[:, :w//2]
    right_half = np.fliplr(mask[:, w//2:])
    min_w = min(left_half.shape[1], right_half.shape[1])
    h_asymmetry = np.sum(left_half[:, :min_w] != right_half[:, :min_w]) / (h * min_w + 1e-6)
    
    top_half = mask[:h//2, :]
    bottom_half = np.flipud(mask[h//2:, :])
    min_h = min(top_half.shape[0], bottom_half.shape[0])
    v_asymmetry = np.sum(top_half[:min_h, :] != bottom_half[:min_h, :]) / (min_h * w + 1e-6)
    
    feats['asymmetry_horizontal'] = float(h_asymmetry)
    feats['asymmetry_vertical'] = float(v_asymmetry)
    
    # Diagonal Asymmetry (Using Hu[0] for parity with original script)
    if cv2 is not None:
        moments = cv2.moments(mask.astype(np.uint8))
        if moments['m00'] > 0:
            hu = cv2.HuMoments(moments).flatten()
            diag_asym = abs(hu[0])
        else:
            diag_asym = 0.0
    else:
        diag_asym = 0.0
    feats['asymmetry_diagonal'] = float(diag_asym)
    feats['asymmetry_overall'] = (h_asymmetry + v_asymmetry) / 2
    
    # --- Border Features (Compactness = P^2 / 4piA) ---
    if cv2 is not None:
         mask_u8 = (mask * 255).astype(np.uint8)
         contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
         if contours:
             cnt = max(contours, key=cv2.contourArea)
             area = cv2.contourArea(cnt)
             perimeter = cv2.arcLength(cnt, True)
             
             feats['border_compactness'] = (perimeter ** 2) / (4 * np.pi * area + 1e-6)
             feats['border_circularity'] = (4 * np.pi * area) / (perimeter ** 2 + 1e-6)
             
             hull = cv2.convexHull(cnt)
             hull_area = cv2.contourArea(hull)
             feats['border_solidity'] = float(area / (hull_area + 1e-6))
             
             hull_perimeter = cv2.arcLength(hull, True)
             feats['border_roughness'] = float(perimeter / (hull_perimeter + 1e-6))
         else:
             for k in ['compactness', 'circularity', 'solidity', 'roughness']: feats[f'border_{k}'] = 0
    else:
        for k in ['compactness', 'circularity', 'solidity', 'roughness']: feats[f'border_{k}'] = 0

    # --- Diameter ---
    rows, cols = np.where(mask)
    if len(rows) > 0:
        major = np.max(rows) - np.min(rows)
        minor = np.max(cols) - np.min(cols)
        area = np.sum(mask)
        feats['diameter_major_axis'] = float(major)
        feats['diameter_minor_axis'] = float(minor)
        feats['diameter_area'] = float(area)
        feats['diameter_equivalent'] = np.sqrt(4 * area / np.pi)
        feats['diameter_aspect_ratio'] = float(major / (minor + 1e-6))
    else:
        for k in ['major_axis', 'minor_axis', 'area', 'equivalent', 'aspect_ratio']: feats[f'diameter_{k}'] = 0

    # --- Color Statistics & Variance ---
    pixels = rgb[mask.astype(bool)]
    if len(pixels) == 0:
        pixels = np.zeros((1, 3))
        
    for i, c in enumerate(['r', 'g', 'b']):
        vals = pixels[:, i]
        feats[f'color_stat_{c}_mean'] = float(np.mean(vals))
        feats[f'color_stat_{c}_std'] = float(np.std(vals))
        feats[f'color_max_{c}'] = float(np.max(vals))
        feats[f'color_min_{c}'] = float(np.min(vals))
        feats[f'color_range_{c}'] = float(np.max(vals) - np.min(vals))
        feats[f'color_variance_{c}'] = float(np.var(vals))
        if 'skew' in globals() or 'skew' in locals():
             feats[f'color_stat_{c}_skew'] = float(skew(vals)) if len(vals) > 1 else 0
             feats[f'color_stat_{c}_kurt'] = float(kurtosis(vals)) if len(vals) > 1 else 0

    if len(pixels) > 1:
        feats['color_correlation_rg'] = float(np.corrcoef(pixels[:,0], pixels[:,1])[0,1])
        feats['color_correlation_rb'] = float(np.corrcoef(pixels[:,0], pixels[:,2])[0,1])
        feats['color_correlation_gb'] = float(np.corrcoef(pixels[:,1], pixels[:,2])[0,1])
    else:
        feats['color_correlation_rg'] = feats['color_correlation_rb'] = feats['color_correlation_gb'] = 0.0
        
    return feats

# ------------------------------------------------------------------
# 3. FEATURE GROUP: WAVELETS (Level 3 - EXACT PARITY)
# ------------------------------------------------------------------
def extract_wavelets(rgb):
    """
    Extract Level 3 Wavelet features from R, G, B channels.
    Matches extract_wavelet_features_rgb.py logic exactly.
    """
    feats = {}
    if pywt is None: return {}

    channels = ['R', 'G', 'B']
    for i, chan in enumerate(channels):
        channel_data = rgb[:, :, i].astype(float) / 255.0 # NORMALIZE to [0,1]
        coeffs = pywt.wavedec2(channel_data, 'db4', level=3)
        
        # Original script: coeffs[0] is LL3
        cA3 = coeffs[0]
        v_list, n_list = compute_subband_stats(cA3, f"{chan}_L3_LL")
        for v, n in zip(v_list, n_list): feats[n] = v
            
        # Detail coeffs from levels 3, 2, 1
        for idx, (cH, cV, cD) in enumerate(coeffs[1:], 1):
            curr_level = 3 - idx + 1
            for sub, band in zip(['LH', 'HL', 'HH'], [cH, cV, cD]):
                v_list, n_list = compute_subband_stats(band, f"{chan}_L{curr_level}_{sub}")
                for v, n in zip(v_list, n_list): feats[n] = v
    return feats

def compute_subband_stats(coeffs, prefix):
    """Matches compute_subband_features in original script."""
    values = []
    names = []
    flat = coeffs.ravel()
    
    # 1. Energy
    values.append(float(np.sum(flat ** 2)))
    names.append(f"{prefix}_energy")
    
    # 2. Mean (absolute)
    values.append(float(np.mean(np.abs(flat))))
    names.append(f"{prefix}_mean")
    
    # 3. Std
    values.append(float(np.std(flat)))
    names.append(f"{prefix}_std")
    
    # 4. Entropy
    try:
        hist, _ = np.histogram(flat, bins=50, density=True)
        hist = hist[hist > 0]
        ent = -np.sum(hist * np.log2(hist + 1e-10))
    except:
        ent = 0.0
    values.append(float(ent))
    names.append(f"{prefix}_entropy")
    
    return values, names

# ------------------------------------------------------------------
# 4. FEATURE GROUP: HU MOMENTS
# ------------------------------------------------------------------
def extract_hu_moments(mask):
    # Same as extract_features_numpy.py
    h, w = mask.shape
    y, x = np.mgrid[:h, :w]
    mask = mask.astype(float)
    m00 = np.sum(mask)
    if m00 == 0: return {f'hu_m{i}': 0.0 for i in range(7)}
    
    m10 = np.sum(x * mask)
    m01 = np.sum(y * mask)
    cx, cy = m10/m00, m01/m00
    
    dx, dy = x - cx, y - cy
    mu20 = np.sum((dx**2)*mask)
    mu02 = np.sum((dy**2)*mask)
    mu11 = np.sum((dx*dy)*mask)
    mu30 = np.sum((dx**3)*mask)
    mu03 = np.sum((dy**3)*mask)
    mu12 = np.sum((dx*dy**2)*mask)
    mu21 = np.sum((dx**2*dy)*mask)
    
    def eta(mu, p, q): return mu / (m00**(1 + (p+q)/2))
    
    n20, n02, n11 = eta(mu20, 2, 0), eta(mu02, 0, 2), eta(mu11, 1, 1)
    n30, n03 = eta(mu30, 3, 0), eta(mu03, 0, 3)
    n12, n21 = eta(mu12, 1, 2), eta(mu21, 2, 1)
    
    h1 = n20 + n02
    h2 = (n20 - n02)**2 + 4 * n11**2
    h3 = (n30 - 3*n12)**2 + (3*n21 - n03)**2
    h4 = (n30 + n12)**2 + (n21 + n03)**2
    h5 = (n30 - 3*n12)*(n30 + n12)*((n30 + n12)**2 - 3*(n21 + n03)**2) + \
         (3*n21 - n03)*(n21 + n03)*(3*(n30 + n12)**2 - (n21 + n03)**2)
    h6 = (n20 - n02)*((n30 + n12)**2 - (n21 + n03)**2) + \
         4*n11*(n30 + n12)*(n21 + n03)
    h7 = (3*n21 - n03)*(n30 + n12)*((n30 + n12)**2 - 3*(n21 + n03)**2) - \
         (n30 - 3*n12)*(n21 + n03)*(3*(n30 + n12)**2 - (n21 + n03)**2)
         
    hus = [h1, h2, h3, h4, h5, h6, h7]
    return {f'hu_m{i}': -np.sign(v)*np.log10(np.abs(v)+1e-30) if v!=0 else 0.0 for i, v in enumerate(hus)}

# ------------------------------------------------------------------
# 5. FEATURE GROUP: CORRELOGRAMS
# ------------------------------------------------------------------
def extract_correlograms(rgb, mask):
    # Quantize
    r, g, b = rgb[...,0]>128, rgb[...,1]>128, rgb[...,2]>128
    q_img = 4*r.astype(np.uint8) + 2*g.astype(np.uint8) + 1*b.astype(np.uint8)
    n_colors = 8
    
    q_masked = q_img.copy()
    q_masked[~mask] = 255 # Ignore background
    
    features = {}
    distances = [1, 3, 5, 7]
    
    for d in distances:
        img_down = np.roll(q_masked, d, axis=0)
        img_right = np.roll(q_masked, d, axis=1)
        
        for c in range(n_colors):
            c_locs = (q_masked == c)
            count = np.sum(c_locs)
            if count == 0:
                features[f'correlo_d{d}_c{c}'] = 0.0
                continue
                
            match_d = (c_locs) & (img_down == c)
            match_d[:d, :] = False
            match_r = (c_locs) & (img_right == c)
            match_r[:, :d] = False
            
            total = np.sum(match_d) + np.sum(match_r)
            features[f'correlo_d{d}_c{c}'] = total / (2.0 * count)
    return features

# ------------------------------------------------------------------
# 6. FEATURE GROUP: C-HOG
# ------------------------------------------------------------------
def extract_chog(rgb):
    feats = {}
    bins = 9
    channels = ['R', 'G', 'B']
    
    for i, chan in enumerate(channels):
        c_data = rgb[:, :, i].astype(float)
        # Pad
        padded = np.pad(c_data, 1, mode='edge')
        dx = padded[1:-1, 2:] - padded[1:-1, :-2]
        dy = padded[2:, 1:-1] - padded[:-2, 1:-1]
        
        mag = np.sqrt(dx**2 + dy**2)
        ang = np.arctan2(dy, dx) * (180/np.pi)
        ang = ang % 180
        
        bin_width = 180 / bins
        hist = np.zeros(bins)
        
        for k in range(bins):
            mask = (ang >= k*bin_width) & (ang < (k+1)*bin_width)
            hist[k] = np.sum(mag[mask])
            
        norm = np.linalg.norm(hist)
        if norm > 0: hist /= norm
        
        for k in range(bins):
            feats[f'chog_{chan}_bin{k}'] = hist[k]
            
    return feats

# ------------------------------------------------------------------
# MASTER EXTRACTION FUNCTION
# ------------------------------------------------------------------
def extract_all_features(image_source, features_json_path):
    """
    1. Load Image
    2. Extract ALL raw features (200+)
    3. Filter and Sort according to features_json_path (Top 50 + C-HOG)
    """
    # Load
    rgb = load_image(image_source)
    mask = get_mask(rgb)
    
    # Extract Raw
    all_feats = {}
    all_feats.update(extract_abcd_color(rgb, mask))
    all_feats.update(extract_wavelets(rgb)) # Be careful: Check Feature names!
    all_feats.update(extract_hu_moments(mask))
    all_feats.update(extract_correlograms(rgb, mask))
    all_feats.update(extract_chog(rgb))
    
    # Load Required Feature List
    if not Path(features_json_path).exists():
        raise FileNotFoundError("Feature list JSON not found")
        
    with open(features_json_path, 'r') as f:
        data = json.load(f)
        # The JSON contains the Top 50 base features
        base_features = data['selected_features']
        
    # We also need the C-HOG features (27 of them)
    # The JSON might NOT contain C-HOG if it was generated before C-HOG integration phase.
    # But our 'chog_model.pkl' EXPECTS [Base 50] + [CHOG 27]
    
    # Let's reconstruct the 27 C-HOG names
    chog_features = []
    for chan in ['R', 'G', 'B']:
        for k in range(9):
            chog_features.append(f'chog_{chan}_bin{k}')
            
    required_features = base_features + chog_features
    
    # Build Feature Vector
    vector = []
    missing = []
    
    for f_name in required_features:
        if f_name in all_feats:
            vector.append(all_feats[f_name])
        else:
            # Fallback for slight naming mismatches (e.g. Wavelet levels)
            # If standard key missing, try to find closest match or append 0
            # Ideally debug this.
            vector.append(0.0)
            missing.append(f_name)
            
    if missing:
        print(f"Warning: Missing features: {missing[:5]}...")
        
    return np.array([vector]), required_features, all_feats

def extract_raw_features_only(image_source):
    """
    Returns a flat dictionary of ALL raw features (ABCD, Wavelets, Hu, Correlo, CHOG).
    Used for creating master training CSVs.
    """
    # Load
    rgb = load_image(image_source)
    mask = get_mask(rgb)
    
    # Extract Raw
    all_feats = {}
    all_feats.update(extract_abcd_color(rgb, mask))
    all_feats.update(extract_wavelets(rgb))
    all_feats.update(extract_hu_moments(mask))
    all_feats.update(extract_correlograms(rgb, mask))
    all_feats.update(extract_chog(rgb))
    
    return all_feats

def extract_raw_features_from_array(rgb, mask=None):
    """
    Extracts features from an ALREADY LOADED numpy array (RGB) and optional mask.
    Used by V2 pipeline (preprocessed in memory).
    """
    if mask is None:
        mask = get_mask(rgb)
    
    # Ensure mask is boolean for proper indexing
    mask = mask.astype(bool)
        
    # Extract Raw
    all_feats = {}
    all_feats.update(extract_abcd_color(rgb, mask))
    all_feats.update(extract_wavelets(rgb))
    all_feats.update(extract_hu_moments(mask))
    all_feats.update(extract_correlograms(rgb, mask))
    all_feats.update(extract_chog(rgb))
    
    return all_feats
