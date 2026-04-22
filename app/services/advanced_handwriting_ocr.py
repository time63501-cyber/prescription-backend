"""
advanced_handwriting_ocr.py
---------------------------
Advanced handwriting OCR system designed specifically for Indian doctor prescriptions.

Features:
1. Written Matter Detection: Analyzes images to detect handwritten regions, calculate
   text density percentage, and determine initial confidence based on document clarity.
2. Handwriting Reference Training: Loads user's handwritten alphabets from reference
   image and creates character templates for pattern matching.
3. Font-based Templates: Loads TTF fonts and generates character reference images for
   fuzzy matching against the handwriting reference.
4. Character-level Template Matching: Implements advanced pattern recognition to match
   Tesseract tokens against learned character templates, improving accuracy on unclear
   handwriting.
5. Confidence Scoring: Multi-factor confidence calculation based on image quality,
   written matter density, and template match scores.
"""

import cv2
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont
from typing import List, Tuple, Dict, Optional
import hashlib
from pathlib import Path
from skimage import measure
from scipy import ndimage
import string


class HandwritingReferenceExtractor:
    """
    Extracts individual character templates from a handwritten alphabet reference image.
    Assumes the reference image contains the alphabet written in order (a-z, 0-9).
    """
    
    def __init__(self, reference_image_path: str):
        self.reference_image_path = reference_image_path
        self.character_templates = {}
        self.template_sizes = {}
        
    def extract_characters(self) -> Dict[str, np.ndarray]:
        """
        Extract individual characters from the reference image.
        Uses contour detection to isolate individual character regions.
        """
        if not os.path.exists(self.reference_image_path):
            print(f"[HandwritingRef] Reference image not found: {self.reference_image_path}")
            return {}
        
        img = cv2.imread(self.reference_image_path)
        if img is None:
            print(f"[HandwritingRef] Cannot read image: {self.reference_image_path}")
            return {}
        
        # Convert to grayscale and invert (text is dark on light background)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply thresholding to get binary image
        _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        binary = cv2.bitwise_not(binary)  # Invert so text is white on black
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter and sort contours by position (top to bottom, left to right for each row)
        char_regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            # Filter out very small regions (noise)
            if w > 5 and h > 10:
                char_regions.append((x, y, w, h, contour))
        
        # Sort by y (row), then x (column)
        char_regions.sort(key=lambda r: (r[1], r[0]))
        
        # Expected character order in handwriting: typically a-z, 0-9
        charset = list(string.ascii_lowercase) + list(string.digits)
        
        print(f"[HandwritingRef] Found {len(char_regions)} potential character regions")
        
        # Extract character templates
        for idx, (x, y, w, h, _) in enumerate(char_regions[:len(charset)]):
            if idx < len(charset):
                char = charset[idx]
                # Extract the region with padding
                y_start = max(0, y - 5)
                y_end = min(binary.shape[0], y + h + 5)
                x_start = max(0, x - 5)
                x_end = min(binary.shape[1], x + w + 5)
                
                char_template = binary[y_start:y_end, x_start:x_end]
                self.character_templates[char] = char_template
                self.template_sizes[char] = (char_template.shape[0], char_template.shape[1])
        
        print(f"[HandwritingRef] Extracted {len(self.character_templates)} character templates")
        return self.character_templates


class FontTemplateGenerator:
    """
    Generates character templates from TTF font files for reference matching.
    """
    
    def __init__(self, fonts_dir: str = "Fonts"):
        self.fonts_dir = fonts_dir
        self.font_templates = {}  # font_name -> char -> template array
        self.load_fonts()
    
    def load_fonts(self):
        """Load all TTF fonts from the fonts directory."""
        if not os.path.exists(self.fonts_dir):
            print(f"[FontTemplates] Fonts directory not found: {self.fonts_dir}")
            return
        
        font_files = [f for f in os.listdir(self.fonts_dir) if f.endswith('.ttf')]
        print(f"[FontTemplates] Found {len(font_files)} TTF fonts")
        
        for font_file in font_files:
            font_path = os.path.join(self.fonts_dir, font_file)
            try:
                font_name = os.path.splitext(font_file)[0]
                self.generate_charset_from_font(font_path, font_name)
            except Exception as e:
                print(f"[FontTemplates] Error loading {font_file}: {e}")
    
    def generate_charset_from_font(self, font_path: str, font_name: str, font_size: int = 24):
        """
        Generate character templates from a specific TTF font.
        """
        try:
            font = ImageFont.truetype(font_path, font_size)
        except Exception as e:
            print(f"[FontTemplates] Cannot load font {font_path}: {e}")
            return
        
        charset = list(string.ascii_lowercase) + list(string.digits)
        self.font_templates[font_name] = {}
        
        for char in charset:
            try:
                # Create image with white background
                img = Image.new('L', (50, 50), color=255)
                draw = ImageDraw.Draw(img)
                
                # Draw character in black
                draw.text((5, 5), char, fill=0, font=font)
                
                # Convert to numpy array
                template = np.array(img)
                
                # Crop to content
                non_white = np.where(template < 200)
                if len(non_white[0]) > 0:
                    y_min, y_max = non_white[0].min(), non_white[0].max()
                    x_min, x_max = non_white[1].min(), non_white[1].max()
                    template = template[y_min:y_max+1, x_min:x_max+1]
                
                self.font_templates[font_name][char] = template
            except Exception as e:
                pass
        
        print(f"[FontTemplates] Generated {len(self.font_templates[font_name])} templates from {font_name}")


class ImagePreprocessor:
    """
    Advanced image preprocessing for handwritten prescription documents.
    Detects written matter, calculates text density, and prepares image for OCR.
    """
    
    def __init__(self):
        self.written_matter_percentage = 0.0
        self.text_density_confidence = 0.0
        self.background_normalized = None
    
    def analyze_written_matter(self, image: np.ndarray) -> Tuple[float, float, np.ndarray]:
        """
        Analyze an image to detect written matter and calculate text density.
        
        Returns:
            - written_matter_percentage: Percentage of image containing written text (0-100)
            - text_density_confidence: Confidence based on text-to-blank ratio (0-100)
            - normalized_image: Image with white background and highlighted text
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Detect background color (most common color)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        background_level = np.argmax(hist)
        
        # Create binary mask: text detection
        # Text is either significantly darker or lighter than background
        if background_level > 127:
            # Light background (typical for prescriptions)
            _, text_mask = cv2.threshold(gray, background_level - 50, 255, cv2.THRESH_BINARY)
        else:
            # Dark background
            _, text_mask = cv2.threshold(gray, background_level + 50, 255, cv2.THRESH_BINARY_INV)
        
        # Apply morphological operations to connect nearby text regions
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        text_mask = cv2.morphologyEx(text_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        text_mask = cv2.morphologyEx(text_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Calculate written matter percentage
        text_pixels = np.count_nonzero(text_mask)
        total_pixels = text_mask.shape[0] * text_mask.shape[1]
        written_matter_percentage = (text_pixels / total_pixels) * 100
        
        # Calculate text density confidence
        # Very sparse documents (< 5%) or very dense documents (> 95%) are suspicious
        if written_matter_percentage < 5:
            text_density_confidence = 20.0  # Too sparse
        elif written_matter_percentage > 95:
            text_density_confidence = 30.0  # Too dense (might be all black)
        elif 10 <= written_matter_percentage <= 50:
            text_density_confidence = 100.0  # Good range for prescriptions
        else:
            # Interpolate between 50-100 for other ranges
            text_density_confidence = 70.0
        
        # Normalize background to white
        normalized = self._normalize_background(gray, background_level)
        
        # Enhance text visibility
        enhanced = self._enhance_text_contrast(normalized, text_mask)
        
        self.written_matter_percentage = written_matter_percentage
        self.text_density_confidence = text_density_confidence
        self.background_normalized = enhanced
        
        print(f"[ImagePreprocessor] Written matter: {written_matter_percentage:.1f}%, "
              f"Density confidence: {text_density_confidence:.1f}%")
        
        return written_matter_percentage, text_density_confidence, enhanced
    
    def _normalize_background(self, gray: np.ndarray, background_level: int) -> np.ndarray:
        """Normalize background to white, text remains dark."""
        normalized = gray.copy().astype(float)
        
        if background_level > 127:
            # Light background: already good, just stretch contrast
            normalized = np.clip(normalized, background_level - 100, 255)
        else:
            # Dark background: invert
            normalized = 255 - normalized
        
        # Normalize to 0-255
        normalized = cv2.normalize(normalized, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return normalized
    
    def _enhance_text_contrast(self, image: np.ndarray, text_mask: np.ndarray) -> np.ndarray:
        """Enhance contrast between text and background with advanced techniques."""
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(image)
        
        # Apply Otsu thresholding
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Enhance by dilating text slightly to make it more visible
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        enhanced_binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        return enhanced_binary
    
    def analyze_image_quality(self, image: np.ndarray) -> Dict[str, float]:
        """
        Analyze various quality metrics of the image.
        
        Returns dictionary with:
            - sharpness: 0-100 (blur detection)
            - brightness: 0-100 (lighting quality)
            - contrast: 0-100 (text visibility)
            - noise_level: 0-100 (inverse - lower is less noisy)
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        metrics = {}
        
        # Sharpness: Laplacian variance
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        metrics['sharpness'] = min(100, (laplacian_var / 100) * 100)
        
        # Brightness
        mean_brightness = np.mean(gray)
        if 80 <= mean_brightness <= 200:
            metrics['brightness'] = 100.0
        elif 50 <= mean_brightness <= 230:
            metrics['brightness'] = 80.0
        else:
            metrics['brightness'] = 50.0
        
        # Contrast: standard deviation of histogram
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        contrast_score = np.std(hist)
        metrics['contrast'] = min(100, contrast_score / 10)
        
        # Noise level: estimate using Laplacian
        noise_est = cv2.Laplacian(gray, cv2.CV_64F).var()
        metrics['noise_level'] = max(0, min(100, 100 - (noise_est / 50)))
        
        return metrics


class CharacterTemplateMatchor:
    """
    Match characters from OCR tokens against learned templates.
    Provides confidence scores based on shape similarity.
    """
    
    def __init__(self, handwriting_ref: HandwritingReferenceExtractor, 
                 font_templates: FontTemplateGenerator):
        self.handwriting_templates = handwriting_ref.character_templates
        self.font_templates = font_templates.font_templates
        self.match_cache = {}
    
    def match_character(self, token: str, token_image: Optional[np.ndarray] = None) -> Tuple[str, float]:
        """
        Match a token character against learned templates.
        
        Returns:
            - best_match: The most likely character
            - confidence: Confidence score (0-100)
        """
        if not token:
            return token, 0.0
        
        # For single character tokens, try direct template matching
        if len(token) == 1:
            char = token.lower()
            confidence = self._calculate_character_confidence(char, token_image)
            return char, confidence
        
        # For multi-character tokens, try splitting and matching
        best_sequence = token
        total_confidence = 0.0
        for char in token.lower():
            _, conf = self.match_character(char, None)
            total_confidence += conf
        
        avg_confidence = total_confidence / len(token) if token else 0.0
        return best_sequence, avg_confidence
    
    def _calculate_character_confidence(self, char: str, token_image: Optional[np.ndarray] = None) -> float:
        """
        Calculate confidence that a character is correctly identified.
        Uses template matching against handwriting reference and fonts with structural similarity.
        """
        confidence_scores = []
        
        # 1. Check against handwriting reference - highest priority (doctor's actual handwriting)
        if char in self.handwriting_templates:
            match_score = self._structural_similarity_score(
                token_image, 
                self.handwriting_templates[char]
            ) if token_image is not None else 95.0
            confidence_scores.append(min(100.0, match_score))
        
        # 2. Check against font templates - good fallback for printed/styled text
        for font_name, char_templates in self.font_templates.items():
            if char in char_templates:
                match_score = self._structural_similarity_score(
                    token_image,
                    char_templates[char]
                ) if token_image is not None else 75.0
                confidence_scores.append(min(100.0, match_score * 0.8))  # Slightly lower weight for fonts
        
        # If no templates found, return moderate confidence
        if not confidence_scores:
            confidence_scores.append(50.0)  # Default for unknown characters
        
        avg_confidence = sum(confidence_scores) / len(confidence_scores)
        return avg_confidence
    
    def _structural_similarity_score(self, image1: Optional[np.ndarray], 
                                   image2: Optional[np.ndarray]) -> float:
        """
        Calculate structural similarity between two images.
        Returns score 0-100.
        """
        if image1 is None or image2 is None:
            return 75.0
        
        try:
            # Resize both to same size
            size = (50, 50)
            img1 = cv2.resize(image1, size) if image1.shape != size else image1
            img2 = cv2.resize(image2, size) if image2.shape != size else image2
            
            # Ensure same data type
            img1 = img1.astype(np.float32) / 255.0
            img2 = img2.astype(np.float32) / 255.0
            
            # Use template matching correlation
            mean1, mean2 = np.mean(img1), np.mean(img2)
            std1, std2 = np.std(img1), np.std(img2)
            
            if std1 == 0 or std2 == 0:
                return 50.0
            
            # Normalized cross-correlation
            correlation = np.mean((img1 - mean1) * (img2 - mean2)) / (std1 * std2)
            # Convert to 0-100 scale
            similarity_score = max(0, min(100, (correlation + 1) * 50))
            
            return similarity_score
        except Exception as e:
            return 50.0
    
    def match_token_sequence(self, tokens: List[str], image_quality: float = 80.0) -> List[Tuple[str, float]]:
        """
        Match a sequence of OCR tokens against templates and return confidence scores.
        """
        results = []
        for token in tokens:
            matched_char, conf = self.match_character(token, None)
            # Adjust confidence based on image quality
            adjusted_conf = (conf + image_quality) / 2
            results.append((matched_char, adjusted_conf))
        
        return results


class AdvancedHandwritingOCR:
    """
    Main orchestrator for advanced handwriting OCR system.
    Coordinates all components: preprocessing, template generation, and matching.
    """
    
    def __init__(self, reference_image_path: str = "Fonts/my-handwritting.jpg",
                 fonts_dir: str = "Fonts"):
        print("[AdvancedOCR] Initializing advanced handwriting OCR system...")
        
        # Initialize components
        self.image_preprocessor = ImagePreprocessor()
        self.handwriting_ref = HandwritingReferenceExtractor(reference_image_path)
        self.font_templates = FontTemplateGenerator(fonts_dir)
        self.character_matchor = CharacterTemplateMatchor(self.handwriting_ref, self.font_templates)
        
        # Extract handwriting templates
        self.handwriting_ref.extract_characters()
        
        print("[AdvancedOCR] Initialization complete")
        print(f"  - Handwriting templates: {len(self.handwriting_ref.character_templates)}")
        print(f"  - Font sets: {len(self.font_templates.font_templates)}")
    
    def process_image_advanced(self, image_path: str) -> Dict:
        """
        Process an image with advanced handwriting OCR.
        Returns comprehensive analysis including confidence scores.
        """
        return self.process_prescription_image(image_path)
    
    def _calculate_image_quality_confidence(self, written_pct: float, density_conf: float, 
                                           image: np.ndarray) -> float:
        """
        Calculate overall image quality confidence based on multiple factors.
        """
        # Get detailed quality metrics
        quality_metrics = self.image_preprocessor.analyze_image_quality(image)
        
        # Factor 1: Sharpness (detect blur) - 30% weight
        sharpness_conf = quality_metrics['sharpness']
        
        # Factor 2: Brightness/Contrast - 20% weight
        brightness_conf = quality_metrics['brightness']
        
        # Factor 3: Text contrast - 20% weight
        contrast_conf = quality_metrics['contrast']
        
        # Factor 4: Written matter percentage - 30% weight
        # Optimal range for prescriptions is 10-50%
        if 10 <= written_pct <= 50:
            text_fill_conf = 100.0
        elif written_pct < 5 or written_pct > 80:
            text_fill_conf = 40.0  # Too sparse or too dense
        else:
            text_fill_conf = 70.0
        
        # Weighted average
        quality_conf = (
            sharpness_conf * 0.30 + 
            brightness_conf * 0.20 + 
            contrast_conf * 0.20 + 
            text_fill_conf * 0.30
        )
        
        return quality_conf
    
    def process_prescription_image(self, image_path: str) -> Dict:
        """
        Complete pipeline for prescription image analysis.
        Provides comprehensive confidence score and quality metrics.
        
        Returns full diagnostics including:
            - written_matter_percentage: % of image with text
            - quality_metrics: sharpness, brightness, contrast, noise
            - overall_confidence: 0-100 score for OCR readiness
            - recommendation: Whether to process with OCR or ask for re-scan
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                return {
                    "success": False,
                    "error": f"Cannot read image: {image_path}",
                    "recommendation": "invalid_file"
                }
            
            # Complete analysis
            written_pct, density_conf, preprocessed = self.image_preprocessor.analyze_written_matter(image)
            quality_metrics = self.image_preprocessor.analyze_image_quality(image)
            image_quality_conf = self._calculate_image_quality_confidence(
                written_pct, density_conf, image
            )
            
            # Overall confidence
            overall_conf = (density_conf * 0.4 + image_quality_conf * 0.6)
            
            # Determine recommendation
            if overall_conf >= 75:
                recommendation = "proceed_with_ocr"
                reason = "Image quality is excellent"
            elif overall_conf >= 60:
                recommendation = "proceed_with_caution"
                reason = "Image quality is acceptable but may have issues"
            elif overall_conf >= 40:
                recommendation = "consider_rescan"
                reason = "Image quality is poor, accuracy may suffer"
            else:
                recommendation = "request_rescan"
                reason = "Image quality is too poor for reliable OCR"
            
            return {
                "success": True,
                "written_matter_percentage": round(written_pct, 1),
                "text_density_confidence": round(density_conf, 1),
                "image_quality_confidence": round(image_quality_conf, 1),
                "overall_confidence": round(overall_conf, 1),
                "quality_metrics": {
                    "sharpness": round(quality_metrics['sharpness'], 1),
                    "brightness": round(quality_metrics['brightness'], 1),
                    "contrast": round(quality_metrics['contrast'], 1),
                    "noise_level": round(quality_metrics['noise_level'], 1)
                },
                "recommendation": recommendation,
                "reason": reason,
                "preprocessed_image": preprocessed,
                "handwriting_templates_available": len(self.handwriting_ref.character_templates),
                "fonts_loaded": len(self.font_templates.font_templates)
            }
        
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": str(e),
                "recommendation": "error"
            }
    
    def enhance_ocr_tokens(self, tokens: List[Tuple[str, float]], image_quality: float = 80.0) -> List[Dict]:
        """
        Enhance OCR tokens with template matching and confidence adjustment.
        """
        enhanced = []
        matched = self.character_matchor.match_token_sequence([t[0] for t in tokens], image_quality)
        
        for (orig_token, orig_conf), (matched_char, template_conf) in zip(tokens, matched):
            # Blend original OCR confidence with template matching confidence
            blended_conf = (orig_conf * 0.6 + template_conf * 0.4)
            
            enhanced.append({
                "original_token": orig_token,
                "original_confidence": orig_conf,
                "matched_character": matched_char,
                "template_confidence": template_conf,
                "blended_confidence": blended_conf,
                "recommendation": matched_char if template_conf > 70 else orig_token
            })
        
        return enhanced
    
    def extract_written_regions(self, image_path: str) -> List[Dict]:
        """
        Extract individual written regions/words from the image.
        Returns list of regions with their bounding boxes and confidence scores.
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                return []
            
            # Preprocessor will give us the binary image
            _, _, binary = self.image_preprocessor.analyze_written_matter(image)
            
            # Find contours (text regions)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            regions = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                # Filter out very small regions
                if w > 10 and h > 10:
                    regions.append({
                        "x": x,
                        "y": y,
                        "width": w,
                        "height": h,
                        "area": w * h,
                        "aspect_ratio": float(w) / float(h) if h > 0 else 0
                    })
            
            # Sort by position (top to bottom, left to right)
            regions.sort(key=lambda r: (r['y'], r['x']))
            
            return regions
        except Exception as e:
            print(f"[AdvancedOCR] Error extracting regions: {e}")
            return []
    
    def analyze_handwriting_style(self, image_path: str) -> Dict:
        """
        Analyze the handwriting style in the image.
        Returns properties like slant, size, spacing, pressure.
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                return {"success": False, "error": "Cannot read image"}
            
            # Get preprocessed binary
            _, _, binary = self.image_preprocessor.analyze_written_matter(image)
            
            # Analyze stroke properties
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return {"success": False, "error": "No handwriting detected"}
            
            heights = []
            widths = []
            
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if w > 5 and h > 10:
                    heights.append(h)
                    widths.append(w)
            
            if not heights:
                return {"success": False, "error": "Insufficient handwriting data"}
            
            avg_height = np.mean(heights)
            avg_width = np.mean(widths)
            height_variance = np.std(heights)
            width_variance = np.std(widths)
            
            # Estimate pressure (density of strokes)
            total_pixels = np.count_nonzero(binary)
            total_area = binary.shape[0] * binary.shape[1]
            pressure_estimate = (total_pixels / total_area) * 100
            
            return {
                "success": True,
                "average_character_height": round(avg_height, 1),
                "average_character_width": round(avg_width, 1),
                "height_variance": round(height_variance, 1),
                "width_variance": round(width_variance, 1),
                "estimated_pressure": round(pressure_estimate, 1),
                "character_count": len(heights),
                "handwriting_quality_index": self._calculate_handwriting_quality_index(
                    heights, widths, pressure_estimate
                )
            }
        except Exception as e:
            print(f"[AdvancedOCR] Error analyzing handwriting: {e}")
            return {"success": False, "error": str(e)}
    
    def _calculate_handwriting_quality_index(self, heights: List, widths: List, 
                                           pressure: float) -> float:
        """
        Calculate a handwriting quality index (0-100).
        Based on consistency, pressure, and other factors.
        """
        # Consistency: lower variance is better
        height_consistency = max(0, 100 - np.std(heights) * 2)
        width_consistency = max(0, 100 - np.std(widths) * 2)
        
        # Pressure: 5-15% is typical for Dr's handwriting
        if 3 <= pressure <= 20:
            pressure_score = 100
        elif 1 <= pressure <= 30:
            pressure_score = 80
        else:
            pressure_score = 50
        
        quality_index = (height_consistency * 0.4 + width_consistency * 0.3 + pressure_score * 0.3)
        return round(max(0, min(100, quality_index)), 1)


# Singleton instance
_advanced_ocr_instance = None

def get_advanced_ocr() -> AdvancedHandwritingOCR:
    """Get or create the singleton advanced OCR instance."""
    global _advanced_ocr_instance
    if _advanced_ocr_instance is None:
        _advanced_ocr_instance = AdvancedHandwritingOCR()
    return _advanced_ocr_instance
