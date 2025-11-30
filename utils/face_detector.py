"""
Face Detection and Alignment using DeepFace
"""
import torch
import numpy as np
from PIL import Image
from deepface import DeepFace
import cv2
from typing import Tuple, Optional, List
import config


class FaceDetector:
    """Face detector using DeepFace with multiple backend options"""
    
    def __init__(self, device='cuda', backend=None):
        """
        Initialize DeepFace detector
        
        Args:
            device: 'cuda' or 'cpu' (Note: DeepFace handles device automatically)
            backend: Face detector backend (if None, uses config)
        """
        self.device = device
        self.backend = backend or config.FACE_DETECTOR_CONFIG.get("backend", "mtcnn")
        self.target_size = config.AGEDB_CONFIG["image_size"]
        
    def detect_and_align(self, image: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
        """
        Detect face and align using DeepFace
        
        Args:
            image: Input image as numpy array (H, W, C) in RGB format
            
        Returns:
            aligned_face: Aligned face image (112, 112, 3)
            landmarks: Facial region dict (converted to array for compatibility)
            prob: Detection confidence (1.0 for DeepFace as it doesn't provide confidence)
            Returns None if no face detected
        """
        try:
            # DeepFace.extract_faces returns a list of face objects
            # Each face object has 'face', 'facial_area', 'confidence'
            faces = DeepFace.extract_faces(
                img_path=image,
                detector_backend=self.backend,
                enforce_detection=True,
                align=True,
                target_size=self.target_size
            )
            
            if not faces or len(faces) == 0:
                return None
            
            # Get the first (best) face
            face_obj = faces[0]
            aligned_face = face_obj['face']
            facial_area = face_obj['facial_area']
            confidence = face_obj.get('confidence', 1.0)
            
            # Convert aligned face from [0, 1] to [0, 255] if needed
            if aligned_face.max() <= 1.0:
                aligned_face = (aligned_face * 255).astype(np.uint8)
            
            # Create landmark array from facial area for compatibility
            # Format: x, y, w, h -> convert to pseudo-landmarks
            landmarks = self._facial_area_to_landmarks(facial_area)
            
            return aligned_face, landmarks, confidence
            
        except Exception as e:
            # If face detection fails, return None
            return None
    
    def _facial_area_to_landmarks(self, facial_area: dict) -> np.ndarray:
        """
        Convert facial area dict to pseudo-landmarks for compatibility
        
        Args:
            facial_area: Dict with keys 'x', 'y', 'w', 'h'
            
        Returns:
            Pseudo-landmarks array (5, 2) with corner points
        """
        x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
        
        # Create 5 pseudo-landmarks from bounding box
        # Format similar to 5-point landmarks: left_eye, right_eye, nose, left_mouth, right_mouth
        landmarks = np.array([
            [x + w * 0.3, y + h * 0.35],  # Left eye approx
            [x + w * 0.7, y + h * 0.35],  # Right eye approx
            [x + w * 0.5, y + h * 0.5],   # Nose approx
            [x + w * 0.35, y + h * 0.75], # Left mouth approx
            [x + w * 0.65, y + h * 0.75]  # Right mouth approx
        ], dtype=np.float32)
        
        return landmarks
    
    def align_face(self, image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """
        Align face using similarity transformation based on landmarks
        Note: With DeepFace, alignment is already done, but keeping this for compatibility
        
        Args:
            image: Input image (H, W, C)
            landmarks: Facial landmarks (5, 2) or None
            
        Returns:
            Aligned face image (112, 112, 3)
        """
        # If image is already the correct size, return as is
        if image.shape[:2] == (112, 112):
            return image
        
        # Otherwise resize
        aligned = cv2.resize(image, (112, 112), interpolation=cv2.INTER_LINEAR)
        return aligned
    
    def batch_detect_align(self, images: List[np.ndarray]) -> List[Optional[Tuple]]:
        """
        Batch process multiple images
        
        Args:
            images: List of input images
            
        Returns:
            List of (aligned_face, landmarks, prob) or None
        """
        results = []
        for image in images:
            result = self.detect_and_align(image)
            results.append(result)
        return results
    
    def quality_check(self, image: np.ndarray, landmarks: np.ndarray, prob: float) -> bool:
        """
        Check if detected face meets quality criteria
        
        Args:
            image: Face image
            landmarks: Facial landmarks
            prob: Detection probability
            
        Returns:
            True if face passes quality check
        """
        # Check detection confidence
        if prob < 0.9:
            return False
        
        # Check if face is too blurry using Laplacian variance
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        if laplacian_var < 100:  # Too blurry
            return False
        
        # Check if landmarks are within image bounds
        h, w = image.shape[:2]
        if np.any(landmarks[:, 0] < 0) or np.any(landmarks[:, 0] >= w):
            return False
        if np.any(landmarks[:, 1] < 0) or np.any(landmarks[:, 1] >= h):
            return False
        
        return True


def test_face_detector():
    """Test face detector on sample images"""
    import os
    from glob import glob
    
    print("Testing FaceDetector with DeepFace backend...")
    detector = FaceDetector(device='cuda' if torch.cuda.is_available() else 'cpu', backend='mtcnn')
    
    # Test on AgeDB-30 images
    image_dir = config.AGEDB_CONFIG["image_folder"]
    if os.path.exists(image_dir):
        image_files = glob(os.path.join(image_dir, "*.jpg"))[:5]
        
        for img_path in image_files:
            print(f"\nProcessing: {os.path.basename(img_path)}")
            
            # Read image
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Detect and align
            result = detector.detect_and_align(img)
            
            if result is not None:
                aligned_face, landmarks, confidence = result
                print(f"  Detection confidence: {confidence:.4f}")
                print(f"  Landmarks shape: {landmarks.shape}")
                print(f"  Aligned face shape: {aligned_face.shape}")
                
                # Check quality
                is_good = detector.quality_check(aligned_face, landmarks, confidence)
                print(f"  Quality check: {'PASS' if is_good else 'FAIL'}")
            else:
                print("  No face detected")
    else:
        print(f"Image directory not found: {image_dir}")
        print("Testing with dummy image...")
        
        # Create a dummy image for testing
        dummy_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        result = detector.detect_and_align(dummy_img)
        if result is None:
            print("No face in dummy image (expected)")
        else:
            print("Face detected in dummy image")


if __name__ == "__main__":
    test_face_detector()
