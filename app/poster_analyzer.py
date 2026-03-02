import numpy as np
from PIL import Image
import io


class PosterAnalyzer:
    """Analyze poster images to extract visual features"""
    
    def __init__(self):
        pass
    
    def analyze_image(self, image_file):
        """
        Analyze uploaded poster image
        
        Args:
            image_file: File-like object or PIL Image
            
        Returns:
            dict: Visual features extracted from poster
        """
        # Load image
        if isinstance(image_file, Image.Image):
            img = image_file
        else:
            img = Image.open(image_file)
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize to reasonable size for faster processing
        img.thumbnail((400, 600), Image.Resampling.LANCZOS)
        
        # Convert to numpy array
        img_array = np.array(img)
        
        # Extract features
        features = {
            'poster_brightness': self._calculate_brightness(img_array),
            'poster_saturation': self._calculate_saturation(img_array),
            'poster_dom_r': None,
            'poster_dom_g': None,
            'poster_dom_b': None,
        }
        
        # Get dominant color
        dom_r, dom_g, dom_b = self._get_dominant_color(img_array)
        features['poster_dom_r'] = dom_r
        features['poster_dom_g'] = dom_g
        features['poster_dom_b'] = dom_b
        
        return features
    
    def _calculate_brightness(self, img_array):
        """Calculate average brightness of image"""
        # Convert to grayscale using luminance formula
        brightness = np.mean(img_array)
        return float(brightness)
    
    def _calculate_saturation(self, img_array):
        """Calculate average saturation of image"""
        # Get R, G, B channels
        r = img_array[:, :, 0].astype(float)
        g = img_array[:, :, 1].astype(float)
        b = img_array[:, :, 2].astype(float)
        
        # Calculate saturation using HSV formula
        max_rgb = np.maximum(np.maximum(r, g), b)
        min_rgb = np.minimum(np.minimum(r, g), b)
        
        # Avoid division by zero
        delta = max_rgb - min_rgb
        saturation = np.zeros_like(max_rgb)
        mask = max_rgb > 0
        saturation[mask] = (delta[mask] / max_rgb[mask]) * 255
        
        return float(np.mean(saturation))
    
    def _get_dominant_color(self, img_array):
        """Get dominant color using median of each channel"""
        # Reshape to list of pixels
        pixels = img_array.reshape(-1, 3)
        
        # Get median color (more robust than mean for dominant color)
        dom_r = float(np.median(pixels[:, 0]))
        dom_g = float(np.median(pixels[:, 1]))
        dom_b = float(np.median(pixels[:, 2]))
        
        return dom_r, dom_g, dom_b
    
    def get_default_features(self):
        """Return default poster features if no image uploaded"""
        return {
            'poster_brightness': 150.0,
            'poster_saturation': 120.0,
            'poster_dom_r': 100.0,
            'poster_dom_g': 100.0,
            'poster_dom_b': 150.0,
        }
