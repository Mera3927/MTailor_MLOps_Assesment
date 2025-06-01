"""
Comprehensive test suite for local model testing.
Tests all functionality before deployment to Cerebrium.
"""

import unittest
import numpy as np
import tempfile
import os
from pathlib import Path
from PIL import Image
import logging
import json
import time

from model import ImagePreprocessor, ONNXClassifier, ImageClassificationPipeline

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestImagePreprocessor(unittest.TestCase):
    """Test cases for ImagePreprocessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.preprocessor = ImagePreprocessor()
        
        # Create test images
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # Create RGB test image
        rgb_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        self.rgb_image_path = self.temp_dir / "test_rgb.jpg"
        Image.fromarray(rgb_image).save(self.rgb_image_path)
        
        # Create RGBA test image
        rgba_image = np.random.randint(0, 255, (100, 100, 4), dtype=np.uint8)
        self.rgba_image_path = self.temp_dir / "test_rgba.png"
        Image.fromarray(rgba_image).save(self.rgba_image_path)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_load_image_success(self):
        """Test successful image loading."""
        image = self.preprocessor.load_image(self.rgb_image_path)
        self.assertEqual(len(image.shape), 3)
        self.assertEqual(image.shape[2], 3)  # RGB channels
    
    def test_load_image_file_not_found(self):
        """Test image loading with non-existent file."""
        with self.assertRaises(FileNotFoundError):
            self.preprocessor.load_image("non_existent_file.jpg")
    
    def test_resize_image(self):
        """Test image resizing."""
        image = np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8)
        resized = self.preprocessor.resize_image(image)
        self.assertEqual(resized.shape, (224, 224, 3))
    
    def test_normalize_image(self):
        """Test image normalization."""
        image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        normalized = self.preprocessor.normalize_image(image)
        
        # Check data type and range
        self.assertEqual(normalized.dtype, np.float32)
        # After normalization, values should be roughly centered around 0
        self.assertTrue(-3 < normalized.mean() < 3)
    
    def test_preprocess_from_path(self):
        """Test complete preprocessing pipeline from file path."""
        processed = self.preprocessor.preprocess(self.rgb_image_path)
        
        # Check output shape: (batch_size, channels, height, width)
        self.assertEqual(processed.shape, (1, 3, 224, 224))
        self.assertEqual(processed.dtype, np.float32)
    
    def test_preprocess_from_array(self):
        """Test preprocessing from numpy array."""
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        processed = self.preprocessor.preprocess(image)
        
        self.assertEqual(processed.shape, (1, 3, 224, 224))
        self.assertEqual(processed.dtype, np.float32)
    
    def test_preprocess_rgba_image(self):
        """Test preprocessing of RGBA image (should convert to RGB)."""
        processed = self.preprocessor.preprocess(self.rgba_image_path)
        self.assertEqual(processed.shape, (1, 3, 224, 224))
    
    def test_invalid_image_shape(self):
        """Test preprocessing with invalid image shape."""
        invalid_image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)  # Grayscale
        with self.assertRaises(ValueError):
            self.preprocessor.preprocess(invalid_image)


class TestONNXClassifier(unittest.TestCase):
    """Test cases for ONNXClassifier class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.onnx_model_path = "resnet18_classifier.onnx"
        
        # Skip tests if ONNX model doesn't exist
        if not Path(self.onnx_model_path).exists():
            self.skipTest(f"ONNX model not found: {self.onnx_model_path}")
        
        self.classifier = ONNXClassifier(self.onnx_model_path)
    
    def test_model_loading(self):
        """Test successful model loading."""
        self.assertIsNotNone(self.classifier.session)
        self.assertIsNotNone(self.classifier.input_name)
        self.assertIsNotNone(self.classifier.output_name)
    
    def test_predict(self):
        """Test basic prediction."""
        # Create random input tensor
        input_tensor = np.random.randint(0, 255, (1, 3, 224, 224), dtype=np.uint8)
        
        output = self.classifier.predict(input_tensor)
        
        # Check output shape and type
        self.assertEqual(output.shape, (1, 1000))  # ImageNet classes
        self.assertEqual(output.dtype, np.float32)
    
    def test_predict_proba(self):
        """Test probability prediction."""
        input_tensor = np.random.randint(0, 255, (1, 3, 224, 224), dtype=np.uint8)
        
        probabilities = self.classifier.predict_proba(input_tensor)
        
        # Check that probabilities sum to 1
        self.assertAlmostEqual(probabilities.sum(), 1.0, places=5)
        self.assertTrue(np.all(probabilities >= 0))
        self.assertTrue(np.all(probabilities <= 1))
    
    def test_predict_class(self):
        """Test class prediction."""
        input_tensor = np.random.randint(0, 255, (1, 3, 224, 224), dtype=np.uint8)
        
        class_id = self.classifier.predict_class(input_tensor)
        
        # Check that class ID is valid
        self.assertIsInstance(class_id, int)
        self.assertTrue(0 <= class_id < 1000)
    
    def test_predict_top_k(self):
        """Test top-k prediction."""
        input_tensor = np.random.randint(0, 255, (1, 3, 224, 224), dtype=np.uint8)
        
        top_5 = self.classifier.predict_top_k(input_tensor, k=5)
        
        # Check output format
        self.assertEqual(len(top_5), 5)
        for class_id, prob in top_5:
            self.assertIsInstance(class_id, int)
            self.assertIsInstance(prob, float)
            self.assertTrue(0 <= class_id < 1000)
            self.assertTrue(0 <= prob <= 1)
        
        # Check that probabilities are in descending order
        probabilities = [prob for _, prob in top_5]
        self.assertEqual(probabilities, sorted(probabilities, reverse=True))
    
    def test_model_not_found(self):
        """Test handling of missing model file."""
        with self.assertRaises(FileNotFoundError):
            ONNXClassifier("non_existent_model.onnx")


class TestImageClassificationPipeline(unittest.TestCase):
    """Test cases for complete classification pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.onnx_model_path = "resnet18_classifier.onnx"
        
        # Skip tests if ONNX model doesn't exist
        if not Path(self.onnx_model_path).exists():
            self.skipTest(f"ONNX model not found: {self.onnx_model_path}")
        
        self.pipeline = ImageClassificationPipeline(self.onnx_model_path)
        
        # Create test image
        self.temp_dir = Path(tempfile.mkdtemp())
        test_image = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
        self.test_image_path = self.temp_dir / "test_image.jpg"
        Image.fromarray(test_image).save(self.test_image_path)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_classify_image_from_path(self):
        """Test image classification from file path."""
        class_id = self.pipeline.classify_image(self.test_image_path)
        
        self.assertIsInstance(class_id, int)
        self.assertTrue(0 <= class_id < 1000)
    
    def test_classify_image_from_array(self):
        """Test image classification from numpy array."""
        image_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        class_id = self.pipeline.classify_image(image_array)
        
        self.assertIsInstance(class_id, int)
        self.assertTrue(0 <= class_id < 1000)
    
    def test_classify_with_probabilities(self):
        """Test image classification with probabilities."""
        predictions = self.pipeline.classify_image_with_probabilities(
            self.test_image_path, top_k=3
        )
        
        self.assertEqual(len(predictions), 3)
        
        for class_id, prob in predictions:
            self.assertIsInstance(class_id, int)
            self.assertIsInstance(prob, float)
            self.assertTrue(0 <= class_id < 1000)
            self.assertTrue(0 <= prob <= 1)
    
    def test_benchmark_inference(self):
        """Test inference benchmarking."""
        stats = self.pipeline.benchmark_inference(self.test_image_path, num_runs=5)
        
        required_keys = ['mean_time', 'std_time', 'min_time', 'max_time', 'num_runs']
        for key in required_keys:
            self.assertIn(key, stats)
            self.assertIsInstance(stats[key], (int, float))
        
        # Check that times are reasonable (should be under 5 seconds per inference)
        self.assertTrue(stats['mean_time'] < 5.0)
        self.assertTrue(stats['min_time'] > 0)
        self.assertEqual(stats['num_runs'], 5)


class TestKnownImages(unittest.TestCase):
    """Test with known images to verify correct classification."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.onnx_model_path = "resnet18_classifier.onnx"
        
        # Skip tests if ONNX model doesn't exist
        if not Path(self.onnx_model_path).exists():
            self.skipTest(f"ONNX model not found: {self.onnx_model_path}")
        
        self.pipeline = ImageClassificationPipeline(self.onnx_model_path)
    
    def test_tench_classification(self):
        """Test classification of tench image (should be class 0)."""
        # This test assumes the tench image is available
        tench_path = "n01440764_tench.JPEG"
        if not Path(tench_path).exists():
            self.skipTest(f"Tench image not found: {tench_path}")
        
        class_id = self.pipeline.classify_image(tench_path)
        
        # Tench should be classified as class 0
        # Allow some tolerance for model variations
        top_5 = self.pipeline.classify_image_with_probabilities(tench_path, top_k=5)
        top_5_classes = [class_id for class_id, _ in top_5]
        
        self.assertIn(0, top_5_classes, "Tench (class 0) should be in top 5 predictions")
    
    def test_mud_turtle_classification(self):
        """Test classification of mud turtle image (should be class 35)."""
        # This test assumes the mud turtle image is available
        turtle_path = "n01667114_mud_turtle.JPEG"
        if not Path(turtle_path).exists():
            self.skipTest(f"Mud turtle image not found: {turtle_path}")
        
        class_id = self.pipeline.classify_image(turtle_path)
        
        # Mud turtle should be classified as class 35
        # Allow some tolerance for model variations
        top_5 = self.pipeline.classify_image_with_probabilities(turtle_path, top_k=5)
        top_5_classes = [class_id for class_id, _ in top_5]
        
        self.assertIn(35, top_5_classes, "Mud turtle (class 35) should be in top 5 predictions")


class TestPerformance(unittest.TestCase):
    """Performance and stress tests."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.onnx_model_path = "resnet18_classifier.onnx"
        
        if not Path(self.onnx_model_path).exists():
            self.skipTest(f"ONNX model not found: {self.onnx_model_path}")
        
        self.pipeline = ImageClassificationPipeline(self.onnx_model_path)
    
    def test_inference_speed(self):
        """Test that inference meets performance requirements (< 3 seconds)."""
        # Create test image
        test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        # Measure inference time
        start_time = time.time()
        _ = self.pipeline.classify_image(test_image)
        inference_time = time.time() - start_time
        
        # Should be faster than 3 seconds as per requirements
        self.assertLess(inference_time, 3.0, 
                       f"Inference took {inference_time:.3f}s, should be < 3s")
        
        logger.info(f"Inference time: {inference_time:.3f}s")
    
    def test_batch_processing(self):
        """Test processing multiple images efficiently."""
        num_images = 10
        test_images = [
            np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            for _ in range(num_images)
        ]
        
        start_time = time.time()
        results = []
        for image in test_images:
            result = self.pipeline.classify_image(image)
            results.append(result)
        total_time = time.time() - start_time
        
        avg_time_per_image = total_time / num_images
        
        self.assertEqual(len(results), num_images)
        self.assertLess(avg_time_per_image, 3.0, 
                       f"Average time per image: {avg_time_per_image:.3f}s")
        
        logger.info(f"Processed {num_images} images in {total_time:.3f}s "
                   f"(avg: {avg_time_per_image:.3f}s per image)")
    
    def test_memory_usage(self):
        """Test that memory usage remains reasonable."""
        import psutil
        import gc
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Process several images
        for _ in range(20):
            test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            _ = self.pipeline.classify_image(test_image)
            gc.collect()  # Force garbage collection
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (< 500MB)
        self.assertLess(memory_increase, 500, 
                       f"Memory increased by {memory_increase:.1f}MB")
        
        logger.info(f"Memory usage: {initial_memory:.1f}MB -> {final_memory:.1f}MB "
                   f"(+{memory_increase:.1f}MB)")


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.onnx_model_path = "resnet18_classifier.onnx"
        
        if not Path(self.onnx_model_path).exists():
            self.skipTest(f"ONNX model not found: {self.onnx_model_path}")
        
        self.pipeline = ImageClassificationPipeline(self.onnx_model_path)
    
    def test_extreme_aspect_ratios(self):
        """Test images with extreme aspect ratios."""
        # Very wide image
        wide_image = np.random.randint(0, 255, (50, 1000, 3), dtype=np.uint8)
        class_id = self.pipeline.classify_image(wide_image)
        self.assertIsInstance(class_id, int)
        
        # Very tall image
        tall_image = np.random.randint(0, 255, (1000, 50, 3), dtype=np.uint8)
        class_id = self.pipeline.classify_image(tall_image)
        self.assertIsInstance(class_id, int)
    
    def test_small_images(self):
        """Test very small images."""
        small_image = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
        class_id = self.pipeline.classify_image(small_image)
        self.assertIsInstance(class_id, int)
    
    def test_large_images(self):
        """Test very large images."""
        # Large image (should be handled by resizing)
        large_image = np.random.randint(0, 255, (2000, 3000, 3), dtype=np.uint8)
        class_id = self.pipeline.classify_image(large_image)
        self.assertIsInstance(class_id, int)
    
    def test_edge_pixel_values(self):
        """Test images with edge pixel values."""
        # All black image
        black_image = np.zeros((224, 224, 3), dtype=np.uint8)
        class_id = self.pipeline.classify_image(black_image)
        self.assertIsInstance(class_id, int)
        
        # All white image
        white_image = np.full((224, 224, 3), 255, dtype=np.uint8)
        class_id = self.pipeline.classify_image(white_image)
        self.assertIsInstance(class_id, int)


def create_test_report(test_results):
    """Create a comprehensive test report."""
    report = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_tests': test_results.testsRun,
        'failures': len(test_results.failures),
        'errors': len(test_results.errors),
        'success_rate': (test_results.testsRun - len(test_results.failures) - len(test_results.errors)) / test_results.testsRun * 100,
        'details': {
            'failures': [str(failure) for failure in test_results.failures],
            'errors': [str(error) for error in test_results.errors]
        }
    }
    
    # Save report to file
    with open('test_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    return report


def main():
    """Run all tests and generate report."""
    logger.info("Starting comprehensive test suite...")
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestImagePreprocessor,
        TestONNXClassifier,
        TestImageClassificationPipeline,
        TestKnownImages,
        TestPerformance,
        TestEdgeCases
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Generate report
    report = create_test_report(result)
    
    # Print summary
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    print(f"Total tests: {report['total_tests']}")
    print(f"Passed: {report['total_tests'] - report['failures'] - report['errors']}")
    print(f"Failed: {report['failures']}")
    print(f"Errors: {report['errors']}")
    print(f"Success rate: {report['success_rate']:.1f}%")
    
    if report['failures'] > 0 or report['errors'] > 0:
        print("\nISSUES FOUND:")
        for failure in report['details']['failures']:
            print(f"FAILURE: {failure}")
        for error in report['details']['errors']:
            print(f"ERROR: {error}")
        return 1
    else:
        print("\nAll tests passed! ✅")
        return 0


if __name__ == "__main__":
    exit_code = main()