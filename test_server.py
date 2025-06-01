#!/usr/bin/env python3
"""
Test script for deployed Cerebrium model.
Tests the deployed model and provides monitoring capabilities.
"""

import argparse
import base64
import io
import json
import logging
import requests
import time
import statistics
from pathlib import Path
from PIL import Image
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Cerebrium API configuration
CEREBRIUM_API_BASE = "https://api.cortex.cerebrium.ai/v4"
API_KEY = "2om0uempl69t4c6fc70ujstsuk"


class CerebriumModelTester:
    """
    Test client for deployed Cerebrium model.
    """
    
    def __init__(self, api_key: str, model_name: str = "resnet18-classifier"):
        """
        Initialize the tester.
        
        Args:
            api_key: Cerebrium API key
            model_name: Name of the deployed model
        """
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = f"{CEREBRIUM_API_BASE}/p/{model_name}/predict"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        logger.info(f"Initialized tester for model: {model_name}")
        logger.info(f"API endpoint: {self.base_url}")
    
    def encode_image_to_base64(self, image_path: Path) -> str:
        """
        Encode image file to base64 string.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Base64 encoded image string
        """
        try:
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Convert to bytes
                buffer = io.BytesIO()
                img.save(buffer, format='JPEG')
                image_bytes = buffer.getvalue()
                
                # Encode to base64
                return base64.b64encode(image_bytes).decode('utf-8')
                
        except Exception as e:
            raise ValueError(f"Failed to encode image {image_path}: {str(e)}")
    
    def predict_image(self, image_path: Path, top_k: int = 5) -> Dict[str, Any]:
        """
        Send prediction request to deployed model.
        
        Args:
            image_path: Path to image file
            top_k: Number of top predictions to return
            
        Returns:
            Prediction response from the model
        """
        logger.info(f"Predicting image: {image_path}")
        
        try:
            # Encode image
            image_base64 = self.encode_image_to_base64(image_path)
            
            # Prepare request
            request_data = {
                "image": image_base64,
                "top_k": top_k
            }
            
            # Send request
            start_time = time.time()
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json=request_data,
                timeout=30  # 30 second timeout
            )
            request_time = time.time() - start_time
            
            # Check response
            response.raise_for_status()
            result = response.json()
            
            # Add timing information
            result['request_time'] = round(request_time, 4)
            
            logger.info(f"Prediction completed in {request_time:.4f}s")
            
            return result
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {str(e)}")
            return {
                "success": False,
                "message": f"Request failed: {str(e)}",
                "request_time": time.time() - start_time if 'start_time' in locals() else 0
            }
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            return {
                "success": False,
                "message": f"Prediction failed: {str(e)}",
                "request_time": 0
            }
    
    def test_known_images(self) -> Dict[str, Any]:
        """
        Test with known images and verify classifications.
        
        Returns:
            Test results dictionary
        """
        logger.info("Testing with known images...")
        
        results = {
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "tests": [],
            "summary": {
                "total": 0,
                "passed": 0,
                "failed": 0
            }
        }
        
        # Test cases: (image_path, expected_class_id, description)
        test_cases = [
            ("n01440764_tench.JPEG", 0, "Tench fish"),
            ("n01667114_mud_turtle.JPEG", 35, "Mud turtle")
        ]
        
        for image_path, expected_class, description in test_cases:
            image_path = Path(image_path)
            
            if not image_path.exists():
                logger.warning(f"Test image not found: {image_path}")
                continue
            
            logger.info(f"Testing {description} (expected class: {expected_class})")
            
            result = self.predict_image(image_path, top_k=5)
            
            test_result = {
                "image": str(image_path),
                "description": description,
                "expected_class": expected_class,
                "result": result
            }
            
            if result.get("success", False):
                predicted_class = result.get("class_id", -1)
                top_5_classes = [pred["class_id"] for pred in result.get("top_predictions", [])]
                
                # Check if expected class is in top 5
                passed = expected_class in top_5_classes
                test_result["passed"] = passed
                test_result["predicted_class"] = predicted_class
                test_result["expected_in_top5"] = passed
                
                if passed:
                    results["summary"]["passed"] += 1
                    logger.info(f"✅ {description}: Expected class {expected_class} found in top 5")
                else:
                    results["summary"]["failed"] += 1
                    logger.warning(f"❌ {description}: Expected class {expected_class} not in top 5. Got: {top_5_classes}")
            else:
                test_result["passed"] = False
                results["summary"]["failed"] += 1
                logger.error(f"❌ {description}: Prediction failed - {result.get('message', 'Unknown error')}")
            
            results["tests"].append(test_result)
            results["summary"]["total"] += 1
            
            # Small delay between requests
            time.sleep(1)
        
        success_rate = (results["summary"]["passed"] / results["summary"]["total"] * 100) if results["summary"]["total"] > 0 else 0
        results["summary"]["success_rate"] = round(success_rate, 1)
        
        logger.info(f"Known images test completed. Success rate: {success_rate:.1f}%")
        
        return results
    
    def test_performance(self, image_path: Path, num_requests: int = 10) -> Dict[str, Any]:
        """
        Test performance and response times of the deployed model.
        
        Args:
            image_path: Path to test image
            num_requests: Number of requests to make for averaging
            
        Returns:
            Performance test results
        """
        logger.info(f"Testing performance with {num_requests} requests...")
        
        if not image_path.exists():
            return {
                "success": False,
                "message": f"Test image not found: {image_path}"
            }
        
        response_times = []
        inference_times = []
        successful_requests = 0
        failed_requests = 0
        
        for i in range(num_requests):
            logger.info(f"Performance test request {i+1}/{num_requests}")
            result = self.predict_image(image_path, top_k=5)
            
            if result.get("success", False):
                successful_requests += 1
                response_times.append(result.get("request_time", 0))
                inference_times.append(result.get("inference_time", 0))
            else:
                failed_requests += 1
                logger.warning(f"Request {i+1} failed: {result.get('message', 'Unknown error')}")
            
            # Small delay between requests to avoid overwhelming the server
            time.sleep(0.5)
        
        if successful_requests == 0:
            return {
                "success": False,
                "message": "All performance test requests failed"
            }
        
        # Calculate statistics
        performance_stats = {
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "num_requests": num_requests,
            "successful_requests": successful_requests,
            "failed_requests": failed_requests,
            "success_rate": (successful_requests / num_requests) * 100,
            "response_time_stats": {
                "mean": statistics.mean(response_times),
                "median": statistics.median(response_times),
                "min": min(response_times),
                "max": max(response_times),
                "std_dev": statistics.stdev(response_times) if len(response_times) > 1 else 0
            },
            "inference_time_stats": {
                "mean": statistics.mean(inference_times),
                "median": statistics.median(inference_times),
                "min": min(inference_times),
                "max": max(inference_times),
                "std_dev": statistics.stdev(inference_times) if len(inference_times) > 1 else 0
            },
            "performance_requirements": {
                "target_response_time": 3.0,
                "meets_requirement": statistics.mean(response_times) < 3.0
            }
        }
        
        logger.info(f"Performance test completed:")
        logger.info(f"  Success rate: {performance_stats['success_rate']:.1f}%")
        logger.info(f"  Average response time: {performance_stats['response_time_stats']['mean']:.3f}s")
        logger.info(f"  Average inference time: {performance_stats['inference_time_stats']['mean']:.3f}s")
        logger.info(f"  Meets performance requirement: {performance_stats['performance_requirements']['meets_requirement']}")
        
        return performance_stats
    
    def test_edge_cases(self) -> Dict[str, Any]:
        """
        Test edge cases and error handling.
        
        Returns:
            Edge case test results
        """
        logger.info("Testing edge cases...")
        
        results = {
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "tests": [],
            "summary": {
                "total": 0,
                "passed": 0,
                "failed": 0
            }
        }
        
        # Test cases
        test_cases = [
            {
                "name": "Invalid base64 data",
                "data": {"image": "invalid_base64_data", "top_k": 5},
                "expect_success": False
            },
            {
                "name": "Missing image field",
                "data": {"top_k": 5},
                "expect_success": False
            },
            {
                "name": "Invalid top_k value",
                "data": {"image": self._create_dummy_base64_image(), "top_k": -1},
                "expect_success": False
            },
            {
                "name": "Large top_k value",
                "data": {"image": self._create_dummy_base64_image(), "top_k": 1001},
                "expect_success": True  # Should handle gracefully
            },
            {
                "name": "Empty request body",
                "data": {},
                "expect_success": False
            }
        ]
        
        for test_case in test_cases:
            logger.info(f"Testing: {test_case['name']}")
            
            try:
                start_time = time.time()
                response = requests.post(
                    self.base_url,
                    headers=self.headers,
                    json=test_case["data"],
                    timeout=10
                )
                request_time = time.time() - start_time
                
                # Parse response
                try:
                    result = response.json()
                except:
                    result = {"success": False, "message": "Invalid JSON response"}
                
                # Determine if test passed
                actual_success = result.get("success", False) and response.status_code == 200
                expected_success = test_case["expect_success"]
                
                test_passed = (actual_success == expected_success)
                
                test_result = {
                    "test_name": test_case["name"],
                    "expected_success": expected_success,
                    "actual_success": actual_success,
                    "status_code": response.status_code,
                    "response": result,
                    "request_time": round(request_time, 4),
                    "passed": test_passed
                }
                
                if test_passed:
                    results["summary"]["passed"] += 1
                    logger.info(f"✅ {test_case['name']}: Behaved as expected")
                else:
                    results["summary"]["failed"] += 1
                    logger.warning(f"❌ {test_case['name']}: Unexpected behavior")
                
            except Exception as e:
                test_result = {
                    "test_name": test_case["name"],
                    "expected_success": test_case["expect_success"],
                    "actual_success": False,
                    "error": str(e),
                    "passed": test_case["expect_success"] == False  # If we expected failure, exception is ok
                }
                
                if test_result["passed"]:
                    results["summary"]["passed"] += 1
                else:
                    results["summary"]["failed"] += 1
                
                logger.error(f"Exception during {test_case['name']}: {str(e)}")
            
            results["tests"].append(test_result)
            results["summary"]["total"] += 1
            
            time.sleep(0.5)  # Small delay between tests
        
        success_rate = (results["summary"]["passed"] / results["summary"]["total"] * 100) if results["summary"]["total"] > 0 else 0
        results["summary"]["success_rate"] = round(success_rate, 1)
        
        logger.info(f"Edge case testing completed. Success rate: {success_rate:.1f}%")
        
        return results
    
    def _create_dummy_base64_image(self) -> str:
        """Create a dummy base64 encoded image for testing."""
        # Create a simple RGB image
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        pil_image = Image.fromarray(image)
        
        # Convert to base64
        buffer = io.BytesIO()
        pil_image.save(buffer, format='JPEG')
        image_bytes = buffer.getvalue()
        
        return base64.b64encode(image_bytes).decode('utf-8')
    
    def test_concurrent_requests(self, image_path: Path, num_concurrent: int = 5) -> Dict[str, Any]:
        """
        Test concurrent request handling.
        
        Args:
            image_path: Path to test image
            num_concurrent: Number of concurrent requests
            
        Returns:
            Concurrent test results
        """
        import threading
        import queue
        
        logger.info(f"Testing {num_concurrent} concurrent requests...")
        
        if not image_path.exists():
            return {
                "success": False,
                "message": f"Test image not found: {image_path}"
            }
        
        results_queue = queue.Queue()
        
        def make_request(request_id):
            """Make a single request and store result."""
            try:
                result = self.predict_image(image_path, top_k=3)
                result["request_id"] = request_id
                results_queue.put(result)
            except Exception as e:
                results_queue.put({
                    "request_id": request_id,
                    "success": False,
                    "message": str(e),
                    "request_time": 0
                })
        
        # Start concurrent requests
        threads = []
        start_time = time.time()
        
        for i in range(num_concurrent):
            thread = threading.Thread(target=make_request, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        total_time = time.time() - start_time
        
        # Collect results
        request_results = []
        successful_requests = 0
        
        while not results_queue.empty():
            result = results_queue.get()
            request_results.append(result)
            if result.get("success", False):
                successful_requests += 1
        
        concurrent_stats = {
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "num_concurrent": num_concurrent,
            "successful_requests": successful_requests,
            "failed_requests": num_concurrent - successful_requests,
            "success_rate": (successful_requests / num_concurrent) * 100,
            "total_time": round(total_time, 4),
            "requests": request_results
        }
        
        logger.info(f"Concurrent test completed:")
        logger.info(f"  {successful_requests}/{num_concurrent} requests successful")
        logger.info(f"  Total time: {total_time:.3f}s")
        logger.info(f"  Success rate: {concurrent_stats['success_rate']:.1f}%")
        
        return concurrent_stats
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the deployed model.
        
        Returns:
            Health check results
        """
        logger.info("Performing health check...")
        
        try:
            # Create a simple test image
            test_image = self._create_dummy_base64_image()
            
            # Make a simple prediction request
            start_time = time.time()
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json={"image": test_image, "top_k": 1},
                timeout=10
            )
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                if result.get("success", False):
                    return {
                        "status": "healthy",
                        "message": "Service is operational",
                        "response_time": round(response_time, 4),
                        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
                    }
                else:
                    return {
                        "status": "unhealthy",
                        "message": f"Service returned error: {result.get('message', 'Unknown error')}",
                        "response_time": round(response_time, 4),
                        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
                    }
            else:
                return {
                    "status": "unhealthy",
                    "message": f"HTTP error: {response.status_code}",
                    "response_time": round(response_time, 4),
                    "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
                }
                
        except Exception as e:
            return {
                "status": "unhealthy",
                "message": f"Health check failed: {str(e)}",
                "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
            }
    
    def comprehensive_test_suite(self, test_image_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Run comprehensive test suite covering all aspects.
        
        Args:
            test_image_path: Optional path to test image (uses default if not provided)
            
        Returns:
            Complete test results
        """
        logger.info("Starting comprehensive test suite...")
        
        # Use default test image if none provided
        if test_image_path is None:
            test_image_path = Path("n01440764_tench.JPEG")
            if not test_image_path.exists():
                # Create a dummy image for testing
                test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                Image.fromarray(test_image).save("temp_test_image.jpg")
                test_image_path = Path("temp_test_image.jpg")
        
        test_results = {
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "test_suite": "comprehensive",
            "tests": {}
        }
        
        # 1. Health Check
        logger.info("=" * 50)
        logger.info("RUNNING HEALTH CHECK")
        logger.info("=" * 50)
        test_results["tests"]["health_check"] = self.health_check()
        
        # 2. Known Images Test
        logger.info("=" * 50)
        logger.info("RUNNING KNOWN IMAGES TEST")
        logger.info("=" * 50)
        test_results["tests"]["known_images"] = self.test_known_images()
        
        # 3. Performance Test
        logger.info("=" * 50)
        logger.info("RUNNING PERFORMANCE TEST")
        logger.info("=" * 50)
        test_results["tests"]["performance"] = self.test_performance(test_image_path, num_requests=5)
        
        # 4. Edge Cases Test
        logger.info("=" * 50)
        logger.info("RUNNING EDGE CASES TEST")
        logger.info("=" * 50)
        test_results["tests"]["edge_cases"] = self.test_edge_cases()
        
        # 5. Concurrent Requests Test
        logger.info("=" * 50)
        logger.info("RUNNING CONCURRENT REQUESTS TEST")
        logger.info("=" * 50)
        test_results["tests"]["concurrent"] = self.test_concurrent_requests(test_image_path, num_concurrent=3)
        
        # Calculate overall summary
        overall_passed = 0
        overall_total = 0
        
        for test_name, test_result in test_results["tests"].items():
            if "summary" in test_result:
                overall_passed += test_result["summary"].get("passed", 0)
                overall_total += test_result["summary"].get("total", 0)
            elif test_name == "health_check":
                overall_total += 1
                if test_result.get("status") == "healthy":
                    overall_passed += 1
        
        test_results["overall_summary"] = {
            "total_tests": overall_total,
            "passed_tests": overall_passed,
            "failed_tests": overall_total - overall_passed,
            "success_rate": (overall_passed / overall_total * 100) if overall_total > 0 else 0
        }
        
        # Clean up temporary image if created
        if test_image_path.name == "temp_test_image.jpg":
            test_image_path.unlink()
        
        logger.info("=" * 50)
        logger.info("COMPREHENSIVE TEST SUITE COMPLETED")
        logger.info("=" * 50)
        logger.info(f"Overall Success Rate: {test_results['overall_summary']['success_rate']:.1f}%")
        logger.info(f"Tests Passed: {test_results['overall_summary']['passed_tests']}/{test_results['overall_summary']['total_tests']}")
        
        return test_results


def save_test_results(results: Dict[str, Any], output_file: str = "test_results.json"):
    """Save test results to JSON file."""
    try:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Test results saved to: {output_file}")
    except Exception as e:
        logger.error(f"Failed to save test results: {str(e)}")


def main():
    """Main entry point for the test script."""
    parser = argparse.ArgumentParser(description="Test deployed Cerebrium model")
    parser.add_argument("--image", type=str, help="Path to test image")
    parser.add_argument("--api-key", type=str, default=API_KEY, help="Cerebrium API key")
    parser.add_argument("--model-name", type=str, default="resnet18-classifier", help="Deployed model name")
    parser.add_argument("--test-type", type=str, choices=[
        "single", "known", "performance", "edge", "concurrent", "health", "comprehensive"
    ], default="comprehensive", help="Type of test to run")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top predictions to return")
    parser.add_argument("--num-requests", type=int, default=10, help="Number of requests for performance testing")
    parser.add_argument("--concurrent", type=int, default=5, help="Number of concurrent requests")
    parser.add_argument("--output", type=str, default="test_results.json", help="Output file for test results")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize tester
    try:
        tester = CerebriumModelTester(args.api_key, args.model_name)
    except Exception as e:
        logger.error(f"Failed to initialize tester: {str(e)}")
        return 1
    
    # Run tests based on type
    try:
        if args.test_type == "single":
            if not args.image:
                logger.error("--image is required for single image testing")
                return 1
            
            image_path = Path(args.image)
            if not image_path.exists():
                logger.error(f"Image not found: {image_path}")
                return 1
            
            result = tester.predict_image(image_path, top_k=args.top_k)
            print(json.dumps(result, indent=2))
            
            if result.get("success", False):
                print(f"\nPredicted class: {result.get('class_id', 'Unknown')}")
                return 0
            else:
                return 1
                
        elif args.test_type == "known":
            results = tester.test_known_images()
            
        elif args.test_type == "performance":
            if not args.image:
                logger.error("--image is required for performance testing")
                return 1
            results = tester.test_performance(Path(args.image), num_requests=args.num_requests)
            
        elif args.test_type == "edge":
            results = tester.test_edge_cases()
            
        elif args.test_type == "concurrent":
            if not args.image:
                logger.error("--image is required for concurrent testing")
                return 1
            results = tester.test_concurrent_requests(Path(args.image), num_concurrent=args.concurrent)
            
        elif args.test_type == "health":
            results = tester.health_check()
            print(json.dumps(results, indent=2))
            return 0 if results.get("status") == "healthy" else 1
            
        elif args.test_type == "comprehensive":
            image_path = Path(args.image) if args.image else None
            results = tester.comprehensive_test_suite(image_path)
        
        # Save and display results
        save_test_results(results, args.output)
        
        # Determine exit code based on results
        if "overall_summary" in results:
            success_rate = results["overall_summary"]["success_rate"]
            return 0 if success_rate >= 80 else 1  # 80% success rate threshold
        elif "summary" in results:
            success_rate = results["summary"]["success_rate"]
            return 0 if success_rate >= 80 else 1
        else:
            return 0
            
    except KeyboardInterrupt:
        logger.info("Testing interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Testing failed: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(main())