"""
PyTorch to ONNX Model Conversion Script
Converts the ResNet18 classification model to ONNX format with embedded preprocessing.
"""

import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
import numpy as np
from pytorch_model import Classifier, BasicBlock
import requests
import os
from typing import Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PreprocessingResNet(nn.Module):
    """
    ResNet model with embedded preprocessing operations.
    This allows us to include preprocessing directly in the ONNX model.
    """
    
    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base_model = base_model
        
        # ImageNet normalization parameters
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with embedded preprocessing:
        1. Assume input is already resized to 224x224 and converted to RGB
        2. Convert to float and normalize to [0, 1]
        3. Apply ImageNet normalization
        4. Run through the base model
        """
        # Convert to float and normalize to [0, 1]
        x = x.float() / 255.0
        
        # Apply ImageNet normalization
        x = (x - self.mean) / self.std
        
        # Forward through base model
        return self.base_model(x)


def download_weights(url: str, output_path: str) -> None:
    """Download model weights from Dropbox URL."""
    logger.info(f"Downloading weights from {url}")
    
    # Convert Dropbox share URL to direct download URL
    direct_url = url.replace('?dl=0', '?dl=1')
    
    response = requests.get(direct_url, stream=True)
    response.raise_for_status()
    
    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    logger.info(f"Weights downloaded to {output_path}")


def load_pytorch_model(weights_path: str) -> nn.Module:
    """Load the PyTorch model with pretrained weights."""
    logger.info("Loading PyTorch model...")
    
    # Create model architecture
    model = Classifier(BasicBlock, [2, 2, 2, 2], num_classes=1000)
    
    # Load weights
    state_dict = torch.load(weights_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    
    logger.info("PyTorch model loaded successfully")
    return model


def convert_to_onnx(
    pytorch_model: nn.Module, 
    onnx_path: str, 
    input_shape: Tuple[int, int, int, int] = (1, 3, 224, 224)
) -> None:
    """
    Convert PyTorch model to ONNX format.
    
    Args:
        pytorch_model: PyTorch model to convert
        onnx_path: Output path for ONNX model
        input_shape: Input tensor shape (batch_size, channels, height, width)
    """
    logger.info("Converting PyTorch model to ONNX...")
    
    # Create dummy input
    dummy_input = torch.randn(*input_shape)
    
    # Export to ONNX
    torch.onnx.export(
        pytorch_model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    logger.info(f"ONNX model saved to {onnx_path}")


def verify_onnx_model(onnx_path: str) -> None:
    """Verify that the ONNX model is valid and can be loaded."""
    logger.info("Verifying ONNX model...")
    
    # Check ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    
    # Test with ONNX Runtime
    ort_session = ort.InferenceSession(onnx_path)
    
    # Test inference
    dummy_input = np.random.randint(0, 255, (1, 3, 224, 224), dtype=np.uint8)
    outputs = ort_session.run(None, {'input': dummy_input})
    
    logger.info(f"ONNX model verification successful. Output shape: {outputs[0].shape}")
    logger.info(f"Output range: [{outputs[0].min():.4f}, {outputs[0].max():.4f}]")


def main():
    """Main conversion pipeline."""
    # Configuration
    weights_url = "https://www.dropbox.com/s/b7641ryzmkceoc9/pytorch_model_weights.pth?dl=0"
    weights_path = "pytorch_model_weights.pth"
    onnx_path = "resnet18_classifier.onnx"
    
    try:
        # Download weights if not present
        if not os.path.exists(weights_path):
            download_weights(weights_url, weights_path)
        
        # Load PyTorch model
        base_model = load_pytorch_model(weights_path)
        
        # Create model with embedded preprocessing
        model_with_preprocessing = PreprocessingResNet(base_model)
        model_with_preprocessing.eval()
        
        # Convert to ONNX
        convert_to_onnx(model_with_preprocessing, onnx_path)
        
        # Verify ONNX model
        verify_onnx_model(onnx_path)
        
        logger.info("Conversion completed successfully!")
        logger.info(f"ONNX model saved as: {onnx_path}")
        logger.info("Model includes embedded preprocessing (normalization)")
        
    except Exception as e:
        logger.error(f"Conversion failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()