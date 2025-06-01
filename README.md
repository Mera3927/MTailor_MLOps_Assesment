# ImageNet Classification Model Deployment on Cerebrium

A production-ready serverless deployment of a ResNet18 image classification model on Cerebrium's GPU platform using Docker containers and ONNX optimization.

## Overview

This project deploys a ResNet18 model trained on ImageNet dataset for image classification. The model accepts 224x224 RGB images and classifies them into one of 1000 ImageNet classes with sub-3-second inference times.

### Key Features

- **ONNX Optimization**: PyTorch model converted to ONNX with embedded preprocessing
- **Serverless Deployment**: Deployed on Cerebrium's GPU platform using custom Docker images
- **Production Ready**: Comprehensive error handling, logging, and monitoring
- **Fast Inference**: Sub-3-second response times with efficient preprocessing pipeline
- **Comprehensive Testing**: Unit tests, integration tests, and deployment validation

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Client App    │───▶│  Cerebrium API   │───▶│  Docker Container│
│                 │    │                  │    │                 │
│ test_server.py  │    │  Base64 Image    │    │     app.py      │
└─────────────────┘    │     Input        │    │       │         │
                       └──────────────────┘    │   model.py      │
                                               │       │         │
                                               │  ONNX Model     │
                                               └─────────────────┘
```

## Project Structure

```
cerebrium-ml-deployment/
├── app.py                          # Cerebrium deployment entry point
├── model.py                        # ONNX inference and preprocessing classes
├── convert_to_onnx.py             # PyTorch to ONNX conversion script
├── test.py                        # Local testing suite
├── test_server.py                 # Deployment testing and monitoring
├── pytorch_model.py               # Original PyTorch model definition
├── Dockerfile                     # Docker container configuration
├── requirements.txt               # Python dependencies
├── cerebrium.toml                 # Cerebrium deployment configuration
├── resnet18_classifier.onnx       # Converted ONNX model (generated)
├── pytorch_model_weights.pth      # PyTorch model weights (downloaded)
├── test_images/                   # Sample test images
│   ├── n01440764_tench.jpg       # Test image (class 0)
│   └── n01667114_mud_turtle.jpg  # Test image (class 35)
└── README.md                      # This file
```

## Prerequisites

- Python 3.8+
- Docker
- Cerebrium account with API key
- Git

## Installation

### 1. Clone Repository

```bash
git clone <your-repository-url>
cd cerebrium-ml-deployment
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Convert Model to ONNX

```bash
python convert_to_onnx.py
```

This script will:
- Download PyTorch model weights from Dropbox
- Load the PyTorch model
- Convert to ONNX format with embedded preprocessing
- Verify the conversion

### 4. Run Local Tests

```bash
python test.py
```

## Deployment

### 1. Configure Cerebrium

Create or update `cerebrium.toml`:

```toml
[cerebrium.deployment]
name = "resnet18-classifier"
python_version = "3.9"
cuda_version = "11.8"
hardware = "GPU"
region = "us-east-1"

[cerebrium.dependencies]
pip = ["requirements.txt"]
```

### 2. Deploy to Cerebrium

```bash
# Login to Cerebrium (if not already done)
cerebrium login

# Deploy the model
cerebrium deploy
```

### 3. Get Deployment Details

After successful deployment, note:
- **API Endpoint URL**: Provided in deployment output
- **API Key**: Your Cerebrium API key

## Usage

### Local Testing

Test all components locally:

```bash
python test.py
```

### API Usage

The deployed model accepts POST requests with the following format:

#### Request Format

```json
{
  "image": "base64_encoded_image_string",
  "top_k": 5
}
```

#### Response Format

```json
{
  "class_id": 285,
  "top_predictions": [
    {"class_id": 285, "probability": 0.8234},
    {"class_id": 281, "probability": 0.1205},
    {"class_id": 287, "probability": 0.0341}
  ],
  "inference_time": 0.1250,
  "total_time": 0.2100,
  "success": true,
  "message": "Classification completed successfully"
}
```

### Testing Deployed Model

Update `test_server.py` with your deployment details:

```python
# Configuration
CEREBRIUM_API_KEY = "your_api_key_here"
CEREBRIUM_ENDPOINT = "https://api.cerebrium.ai/v1/deployments/your-deployment-id/predict"
```

Run deployment tests:

```bash
# Test with single image
python test_server.py --image test_images/n01440764_tench.jpg

# Run comprehensive tests
python test_server.py --run-tests

# Monitor deployment health
python test_server.py --monitor
```

## API Endpoints

### Main Prediction Endpoint

- **URL**: `/predict`
- **Method**: POST
- **Input**: Base64 encoded image
- **Output**: Classification results

### Health Check

- **URL**: `/health`
- **Method**: GET
- **Output**: Service health status

### Model Information

- **URL**: `/info`
- **Method**: GET
- **Output**: Model metadata and configuration

## Development

### Running Local Development Server

```bash
python app.py
```

### Adding New Tests

Add test cases to `test.py`:

```python
def test_custom_case(self):
    """Add your custom test case."""
    # Your test implementation
    pass
```

### Modifying Preprocessing

Update preprocessing in `model.py` → `ImagePreprocessor` class:

```python
def custom_preprocessing(self, image):
    # Your custom preprocessing steps
    return processed_image
```

## Performance Optimization

### Model Optimization

- **ONNX Runtime**: Optimized inference engine
- **Embedded Preprocessing**: Reduces data transfer overhead
- **GPU Acceleration**: CUDA-enabled inference when available

### Expected Performance

- **Inference Time**: < 200ms
- **Total Response Time**: < 3 seconds
- **Throughput**: ~5-10 requests/second
- **Memory Usage**: ~2GB GPU memory

## Monitoring and Logging

### Application Logs

Logs are automatically collected by Cerebrium:

```python
import logging
logger = logging.getLogger(__name__)
logger.info("Your log message")
```

### Health Monitoring

Monitor deployment health:

```bash
python test_server.py --monitor --interval 60
```

### Performance Metrics

Track key metrics:
- Inference latency
- Request success rate
- Error patterns
- Resource utilization

## Troubleshooting

### Common Issues

#### 1. Model Loading Errors

```
Error: Model not found: resnet18_classifier.onnx
```

**Solution**: Run `python convert_to_onnx.py` to generate the ONNX model.

#### 2. Image Format Errors

```
Error: Failed to decode base64 image
```

**Solution**: Ensure image is properly base64 encoded:

```python
import base64
with open("image.jpg", "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode('utf-8')
```

#### 3. Deployment Timeout

```
Error: Deployment timeout
```

**Solution**: 
- Check Docker image size (should be < 10GB)
- Verify all dependencies in requirements.txt
- Check Cerebrium logs for detailed error messages

#### 4. Inference Slow Performance

**Solution**:
- Ensure GPU hardware is selected in cerebrium.toml
- Check if CUDA providers are available
- Monitor resource utilization

### Debug Mode

Enable detailed logging:

```python
logging.basicConfig(level=logging.DEBUG)
```

### Support

For deployment issues:
1. Check Cerebrium dashboard logs
2. Run local tests with `python test.py`
3. Verify model conversion with `python convert_to_onnx.py`
4. Test API locally with `python app.py`

## Contributing

### Code Style

- Follow PEP 8 guidelines
- Use type hints
- Add comprehensive docstrings
- Write unit tests for new features

### Testing

Before submitting changes:

```bash
# Run all tests
python test.py

# Run deployment tests
python test_server.py --run-tests

# Check code formatting
black *.py

# Check imports
isort *.py
```

### Git Workflow

```bash
# Create feature branch
git checkout -b feature/your-feature

# Make changes and commit
git add .
git commit -m "feat: add your feature description"

# Push and create PR
git push origin feature/your-feature
```

## License

This project is licensed under the MIT License.

## Acknowledgments

- ImageNet dataset and classification classes
- PyTorch and ONNX communities
- Cerebrium platform for serverless ML deployment

---

**Note**: Replace placeholder values (API keys, endpoints, repository URLs) with your actual deployment details before using this README.
