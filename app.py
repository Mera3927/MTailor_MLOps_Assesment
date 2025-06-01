"""
Cerebrium deployment application.
Main entry point for the serverless ML model deployment.
"""

import base64
import io
import logging
import time
from typing import Dict, Any, List, Tuple
import numpy as np
from PIL import Image

from model import ImageClassificationPipeline

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model pipeline - initialized once when container starts
model_pipeline = None


def load_model():
    """
    Load the model pipeline. Called once during container initialization.
    """
    global model_pipeline
    
    try:
        logger.info("Loading ONNX model pipeline...")
        model_pipeline = ImageClassificationPipeline("resnet18_classifier.onnx")
        logger.info("Model pipeline loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise


def decode_base64_image(base64_string: str) -> np.ndarray:
    """
    Decode base64 encoded image to numpy array.
    
    Args:
        base64_string: Base64 encoded image string
        
    Returns:
        Image as numpy array
        
    Raises:
        ValueError: If image cannot be decoded
    """
    try:
        # Remove data URL prefix if present
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        # Decode base64
        image_data = base64.b64decode(base64_string)
        
        # Convert to PIL Image
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array
        return np.array(image)
        
    except Exception as e:
        raise ValueError(f"Failed to decode base64 image: {str(e)}")


def validate_input(data: Dict[str, Any]) -> None:
    """
    Validate input data format.
    
    Args:
        data: Input data dictionary
        
    Raises:
        ValueError: If input format is invalid
    """
    if not isinstance(data, dict):
        raise ValueError("Input must be a dictionary")
    
    if 'image' not in data:
        raise ValueError("Missing 'image' field in input")
    
    if not isinstance(data['image'], str):
        raise ValueError("Image must be a base64 encoded string")


def predict(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main prediction function called by Cerebrium.
    
    Args:
        item: Input data containing base64 encoded image
        
    Returns:
        Prediction results dictionary
        
    Expected input format:
    {
        "image": "base64_encoded_image_string",
        "top_k": 5  # optional, defaults to 5
    }
    
    Returns:
    {
        "class_id": int,
        "top_predictions": [{"class_id": int, "probability": float}, ...],
        "inference_time": float,
        "success": bool,
        "message": str
    }
    """
    start_time = time.time()
    
    try:
        # Validate input
        validate_input(item)
        
        # Get parameters
        base64_image = item['image']
        top_k = item.get('top_k', 5)
        
        logger.info(f"Processing prediction request with top_k={top_k}")
        
        # Decode image
        image_array = decode_base64_image(base64_image)
        logger.info(f"Decoded image shape: {image_array.shape}")
        
        # Ensure model is loaded
        if model_pipeline is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Get predictions
        prediction_start = time.time()
        
        # Get top class
        class_id = model_pipeline.classify_image(image_array)
        
        # Get top-k predictions with probabilities
        top_predictions = model_pipeline.classify_image_with_probabilities(
            image_array, top_k=top_k
        )
        
        prediction_time = time.time() - prediction_start
        total_time = time.time() - start_time
        
        # Format results
        result = {
            "class_id": int(class_id),
            "top_predictions": [
                {"class_id": int(cid), "probability": float(prob)}
                for cid, prob in top_predictions
            ],
            "inference_time": round(prediction_time, 4),
            "total_time": round(total_time, 4),
            "success": True,
            "message": "Classification completed successfully"
        }
        
        logger.info(f"Prediction completed in {total_time:.4f}s (inference: {prediction_time:.4f}s)")
        logger.info(f"Predicted class: {class_id}")
        
        return result
        
    except Exception as e:
        error_time = time.time() - start_time
        error_message = str(e)
        
        logger.error(f"Prediction failed after {error_time:.4f}s: {error_message}")
        
        return {
            "class_id": -1,
            "top_predictions": [],
            "inference_time": 0.0,
            "total_time": round(error_time, 4),
            "success": False,
            "message": f"Prediction failed: {error_message}"
        }


def health_check() -> Dict[str, Any]:
    """
    Health check endpoint for monitoring.
    
    Returns:
        Health status dictionary
    """
    try:
        # Check if model is loaded
        if model_pipeline is None:
            return {
                "status": "unhealthy",
                "message": "Model not loaded",
                "timestamp": time.time()
            }
        
        # Test with dummy image
        dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        start_time = time.time()
        _ = model_pipeline.classify_image(dummy_image)
        inference_time = time.time() - start_time
        
        return {
            "status": "healthy",
            "message": "Service is operational",
            "model_loaded": True,
            "test_inference_time": round(inference_time, 4),
            "timestamp": time.time()
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "message": f"Health check failed: {str(e)}",
            "timestamp": time.time()
        }


def get_model_info() -> Dict[str, Any]:
    """
    Get information about the loaded model.
    
    Returns:
        Model information dictionary
    """
    try:
        if model_pipeline is None:
            return {
                "model_loaded": False,
                "message": "Model not loaded"
            }
        
        return {
            "model_loaded": True,
            "model_type": "ResNet18 ImageNet Classifier",
            "input_size": "224x224x3",
            "num_classes": 1000,
            "preprocessing_embedded": True,
            "supported_formats": ["RGB images"],
            "expected_inference_time": "< 3 seconds"
        }
        
    except Exception as e:
        return {
            "model_loaded": False,
            "message": f"Error getting model info: {str(e)}"
        }


# Initialize model when module is imported
try:
    load_model()
    logger.info("Cerebrium app initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Cerebrium app: {str(e)}")
    # Don't raise here to allow container to start - errors will be handled in predict()


if __name__ == "__main__":
    # Test the application locally
    print("Testing Cerebrium app locally...")
    
    # Test health check
    health = health_check()
    print(f"Health check: {health}")
    
    # Test model info
    info = get_model_info()
    print(f"Model info: {info}")
    
    # Test with dummy image
    dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    pil_image = Image.fromarray(dummy_image)
    
    # Convert to base64
    buffer = io.BytesIO()
    pil_image.save(buffer, format='JPEG')
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    # Test prediction
    test_item = {
        "image": image_base64,
        "top_k": 3
    }
    
    result = predict(test_item)
    print(f"Test prediction result: {result}")