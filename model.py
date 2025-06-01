"""
Model classes for ONNX inference and image preprocessing.
Contains modular classes for production deployment.
"""

import numpy as np
import onnxruntime as ort
from PIL import Image
from typing import Union, Tuple, List, Optional
import cv2
import logging
import time
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """
    Image preprocessing class for ResNet model.
    Handles all preprocessing steps required for ImageNet-trained models.
    """
    
    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        """
        Initialize preprocessor with target image size.
        
        Args:
            target_size: Target image dimensions (height, width)
        """
        self.target_size = target_size
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
        logger.info(f"ImagePreprocessor initialized with target_size: {target_size}")
    
    def load_image(self, image_path: Union[str, Path]) -> np.ndarray:
        """
        Load image from file path.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Loaded image as numpy array
            
        Raises:
            FileNotFoundError: If image file doesn't exist
            ValueError: If image cannot be loaded
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        try:
            image = Image.open(image_path)
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            return np.array(image)
        except Exception as e:
            raise ValueError(f"Failed to load image {image_path}: {str(e)}")
    
    def resize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Resize image to target size using bilinear interpolation.
        
        Args:
            image: Input image array
            
        Returns:
            Resized image array
        """
        if len(image.shape) != 3:
            raise ValueError(f"Expected 3D image array, got shape: {image.shape}")
        
        # Use cv2 for bilinear interpolation (faster than PIL)
        resized = cv2.resize(image, self.target_size, interpolation=cv2.INTER_LINEAR)
        return resized
    
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image using ImageNet statistics.
        
        Args:
            image: Input image array (0-255 range)
            
        Returns:
            Normalized image array
        """
        # Convert to float and normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Apply ImageNet normalization
        image = (image - self.mean) / self.std
        
        return image
    
    def preprocess(self, image_input: Union[str, Path, np.ndarray]) -> np.ndarray:
        """
        Complete preprocessing pipeline.
        
        Args:
            image_input: Image file path or numpy array
            
        Returns:
            Preprocessed image ready for model inference
        """
        start_time = time.time()
        
        # Load image if path is provided
        if isinstance(image_input, (str, Path)):
            image = self.load_image(image_input)
        else:
            image = image_input.copy()
        
        # Ensure RGB format
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Already RGB
            pass
        elif len(image.shape) == 3 and image.shape[2] == 4:
            # RGBA to RGB
            image = image[:, :, :3]
        else:
            raise ValueError(f"Unsupported image shape: {image.shape}")
        
        # Resize to target size
        image = self.resize_image(image)
        
        # Normalize
        image = self.normalize_image(image)
        
        # Add batch dimension and transpose to CHW format
        image = np.transpose(image, (2, 0, 1))  # HWC to CHW
        image = np.expand_dims(image, axis=0)   # Add batch dimension
        
        preprocessing_time = time.time() - start_time
        logger.debug(f"Preprocessing completed in {preprocessing_time:.3f}s")
        
        return image


class ONNXClassifier:
    """
    ONNX model wrapper for image classification.
    Handles model loading, inference, and result processing.
    """
    
    def __init__(self, model_path: Union[str, Path], num_classes: int = 1000):
        """
        Initialize ONNX classifier.
        
        Args:
            model_path: Path to ONNX model file
            num_classes: Number of output classes
        """
        self.model_path = Path(model_path)
        self.num_classes = num_classes
        self.session = None
        self.input_name = None
        self.output_name = None
        
        self._load_model()
        logger.info(f"ONNXClassifier initialized with model: {self.model_path}")
    
    def _load_model(self) -> None:
        """Load ONNX model and initialize inference session."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {self.model_path}")
        
        try:
            # Create inference session with optimizations
            providers = ['CPUExecutionProvider']
            if ort.get_available_providers():
                # Use GPU if available
                available_providers = ort.get_available_providers()
                if 'CUDAExecutionProvider' in available_providers:
                    providers.insert(0, 'CUDAExecutionProvider')
            
            self.session = ort.InferenceSession(
                str(self.model_path),
                providers=providers
            )
            
            # Get input/output names
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            
            logger.info(f"Model loaded successfully. Using providers: {self.session.get_providers()}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load ONNX model: {str(e)}")
    
    def predict(self, input_tensor: np.ndarray) -> np.ndarray:
        """
        Run inference on input tensor.
        
        Args:
            input_tensor: Preprocessed input tensor
            
        Returns:
            Raw model output (logits)
        """
        if self.session is None:
            raise RuntimeError("Model not loaded. Call _load_model() first.")
        
        start_time = time.time()
        
        try:
            # Run inference
            outputs = self.session.run(
                [self.output_name],
                {self.input_name: input_tensor}
            )
            
            inference_time = time.time() - start_time
            logger.debug(f"Inference completed in {inference_time:.3f}s")
            
            return outputs[0]
            
        except Exception as e:
            raise RuntimeError(f"Inference failed: {str(e)}")
    
    def predict_proba(self, input_tensor: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities using softmax.
        
        Args:
            input_tensor: Preprocessed input tensor
            
        Returns:
            Softmax probabilities
        """
        logits = self.predict(input_tensor)
        
        # Apply softmax to convert logits to probabilities
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        return probabilities
    
    def predict_class(self, input_tensor: np.ndarray) -> int:
        """
        Get predicted class ID.
        
        Args:
            input_tensor: Preprocessed input tensor
            
        Returns:
            Predicted class ID
        """
        logits = self.predict(input_tensor)
        return int(np.argmax(logits, axis=1)[0])
    
    def predict_top_k(self, input_tensor: np.ndarray, k: int = 5) -> List[Tuple[int, float]]:
        """
        Get top-k predictions with probabilities.
        
        Args:
            input_tensor: Preprocessed input tensor
            k: Number of top predictions to return
            
        Returns:
            List of (class_id, probability) tuples
        """
        probabilities = self.predict_proba(input_tensor)
        
        # Get top-k indices
        top_k_indices = np.argsort(probabilities[0])[::-1][:k]
        
        # Return class IDs and probabilities
        return [(int(idx), float(probabilities[0][idx])) for idx in top_k_indices]


class ImageClassificationPipeline:
    """
    Complete image classification pipeline combining preprocessing and inference.
    """
    
    def __init__(self, model_path: Union[str, Path], target_size: Tuple[int, int] = (224, 224)):
        """
        Initialize classification pipeline.
        
        Args:
            model_path: Path to ONNX model
            target_size: Target image size for preprocessing
        """
        self.preprocessor = ImagePreprocessor(target_size=target_size)
        self.classifier = ONNXClassifier(model_path)
        logger.info("ImageClassificationPipeline initialized")
    
    def classify_image(self, image_input: Union[str, Path, np.ndarray]) -> int:
        """
        Classify single image and return class ID.
        
        Args:
            image_input: Image file path or numpy array
            
        Returns:
            Predicted class ID
        """
        # Preprocess image
        processed_image = self.preprocessor.preprocess(image_input)
        
        # Classify
        class_id = self.classifier.predict_class(processed_image)
        
        return class_id
    
    def classify_image_with_probabilities(
        self, 
        image_input: Union[str, Path, np.ndarray], 
        top_k: int = 5
    ) -> List[Tuple[int, float]]:
        """
        Classify image and return top-k predictions with probabilities.
        
        Args:
            image_input: Image file path or numpy array
            top_k: Number of top predictions to return
            
        Returns:
            List of (class_id, probability) tuples
        """
        # Preprocess image
        processed_image = self.preprocessor.preprocess(image_input)
        
        # Get top-k predictions
        predictions = self.classifier.predict_top_k(processed_image, k=top_k)
        
        return predictions
    
    def benchmark_inference(self, image_input: Union[str, Path, np.ndarray], num_runs: int = 10) -> dict:
        """
        Benchmark inference performance.
        
        Args:
            image_input: Image file path or numpy array
            num_runs: Number of inference runs for averaging
            
        Returns:
            Performance statistics
        """
        # Preprocess once
        processed_image = self.preprocessor.preprocess(image_input)
        
        # Run multiple inferences
        times = []
        for _ in range(num_runs):
            start_time = time.time()
            _ = self.classifier.predict(processed_image)
            end_time = time.time()
            times.append(end_time - start_time)
        
        return {
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'num_runs': num_runs
        }