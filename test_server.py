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
from pathlib import Path
from PIL import Image
import numpy as np
from typing import Dict, Any, List, Optional

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
        
        logger.info(f"Known images test completed. Success rate: {success_rate}%")
        
        return results
    
    def test_model_health(self) -> Dict[str, Any]:
        """
        Test model health and response times.
        
        Returns:
            Health test results
        """
        logger.info("Testing model health...")
        
        results = {
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "health_checks": [],
            "performance_metrics": {
                "avg_response_time": 0,
                "min_response_time": float('inf'),
                "max_response_time": 0,
                "success_count": 0,
                "total_requests": 0
            }
        }
        
        # Use the tench image for health checks if available
        test_image = Path("n01440764_tench.JPEG")
        if not test_image.exists():
            logger.error("Test image not found for health check")
            return results
        
        # Perform multiple requests to test consistency
        num_requests = 5
        response_times = []
        
        for i in range(num_requests):
            logger.info(f"Health check request {i+1}/{num_requests}")
            
            result = self.predict_image(test_image, top_k=1)
            
            health_check = {
                "request_number": i + 1,
                "success": result.get("success", False),
                "response_time": result.get("request_time", 0),
                "timestamp": time.strftime('%H:%M:%S')
            }
            
            results["health_checks"].append(health_check)
            results["performance_metrics"]["total_requests"] += 1
            
            if health_check["success"]:
                results["performance_metrics"]["success_count"] += 1
                response_time = health_check["response_time"]
                response_times.append(response_time)
                
                # Update min/max response times
                if response_time < results["performance_metrics"]["min_response_time"]:
                    results["performance_metrics"]["min_response_time"] = response_time
                if response_time > results["performance_metrics"]["max_response_time"]:
                    results["performance_metrics"]["max_response_time"] = response_time
            
            # Small delay between requests
            time.sleep(2)
        
        # Calculate average response time
        if response_times:
            results["performance_metrics"]["avg_response_time"] = round(
                sum(response_times) / len(response_times), 4
            )
            results["performance_metrics"]["min_response_time"] = round(
                results["performance_metrics"]["min_response_time"], 4
            )
            results["performance_metrics"]["max_response_time"] = round(
                results["performance_metrics"]["max_response_time"], 4
            )
        else:
            results["performance_metrics"]["min_response_time"] = 0
        
        # Calculate success rate
        success_rate = (results["performance_metrics"]["success_count"] / 
                       results["performance_metrics"]["total_requests"] * 100)
        results["performance_metrics"]["success_rate"] = round(success_rate, 1)
        
        logger.info(f"Health check completed. Success rate: {success_rate}%")
        logger.info(f"Average response time: {results['performance_metrics']['avg_response_time']}s")
        
        return results
    
    def test_edge_cases(self) -> Dict[str, Any]:
        """
        Test edge cases and error handling.
        
        Returns:
            Edge case test results
        """
        logger.info("Testing edge cases...")
        
        results = {
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "edge_tests": []
        }
        
        # Test 1: Invalid base64 data
        logger.info("Testing invalid base64 data...")
        try:
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json={"image": "invalid_base64_data", "top_k": 5},
                timeout=30
            )
            
            edge_test = {
                "test": "invalid_base64",
                "description": "Test with invalid base64 data",
                "status_code": response.status_code,
                "success": response.status_code >= 400,  # Should fail gracefully
                "response": response.json() if response.headers.get('content-type', '').startswith('application/json') else response.text
            }
        except Exception as e:
            edge_test = {
                "test": "invalid_base64",
                "description": "Test with invalid base64 data",
                "success": True,  # Exception is expected
                "error": str(e)
            }
        
        results["edge_tests"].append(edge_test)
        
        # Test 2: Missing required fields
        logger.info("Testing missing required fields...")
        try:
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json={"top_k": 5},  # Missing image field
                timeout=30
            )
            
            edge_test = {
                "test": "missing_image_field",
                "description": "Test with missing image field",
                "status_code": response.status_code,
                "success": response.status_code >= 400,  # Should fail gracefully
                "response": response.json() if response.headers.get('content-type', '').startswith('application/json') else response.text
            }
        except Exception as e:
            edge_test = {
                "test": "missing_image_field",
                "description": "Test with missing image field",
                "success": True,  # Exception is expected
                "error": str(e)
            }
        
        results["edge_tests"].append(edge_test)
        
        # Test 3: Invalid top_k values
        logger.info("Testing invalid top_k values...")
        test_image = Path("n01440764_tench.JPEG")
        if test_image.exists():
            invalid_top_k_values = [-1, 0, 1001]  # Invalid values
            
            for top_k in invalid_top_k_values:
                try:
                    image_base64 = self.encode_image_to_base64(test_image)
                    response = requests.post(
                        self.base_url,
                        headers=self.headers,
                        json={"image": image_base64, "top_k": top_k},
                        timeout=30
                    )
                    
                    edge_test = {
                        "test": f"invalid_top_k_{top_k}",
                        "description": f"Test with invalid top_k value: {top_k}",
                        "status_code": response.status_code,
                        "success": response.status_code < 500,  # Should handle gracefully
                        "response": response.json() if response.headers.get('content-type', '').startswith('application/json') else response.text
                    }
                except Exception as e:
                    edge_test = {
                        "test": f"invalid_top_k_{top_k}",
                        "description": f"Test with invalid top_k value: {top_k}",
                        "success": False,
                        "error": str(e)
                    }
                
                results["edge_tests"].append(edge_test)
        
        logger.info("Edge case testing completed")
        
        return results
    
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """
        Run all tests and generate comprehensive report.
        
        Returns:
            Comprehensive test results
        """
        logger.info("Starting comprehensive test suite...")
        
        comprehensive_results = {
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "model_name": self.model_name,
            "api_endpoint": self.base_url,
            "test_results": {},
            "overall_summary": {}
        }
        
        # Run known images test
        try:
            comprehensive_results["test_results"]["known_images"] = self.test_known_images()
        except Exception as e:
            logger.error(f"Known images test failed: {str(e)}")
            comprehensive_results["test_results"]["known_images"] = {"error": str(e)}
        
        # Run health test
        try:
            comprehensive_results["test_results"]["health_check"] = self.test_model_health()
        except Exception as e:
            logger.error(f"Health check test failed: {str(e)}")
            comprehensive_results["test_results"]["health_check"] = {"error": str(e)}
        
        # Run edge cases test
        try:
            comprehensive_results["test_results"]["edge_cases"] = self.test_edge_cases()
        except Exception as e:
            logger.error(f"Edge cases test failed: {str(e)}")
            comprehensive_results["test_results"]["edge_cases"] = {"error": str(e)}
        
        # Calculate overall summary
        total_tests = 0
        passed_tests = 0
        
        # Count known images tests
        known_images_results = comprehensive_results["test_results"].get("known_images", {})
        if "summary" in known_images_results:
            total_tests += known_images_results["summary"].get("total", 0)
            passed_tests += known_images_results["summary"].get("passed", 0)
        
        # Count health check tests
        health_results = comprehensive_results["test_results"].get("health_check", {})
        if "performance_metrics" in health_results:
            total_tests += health_results["performance_metrics"].get("total_requests", 0)
            passed_tests += health_results["performance_metrics"].get("success_count", 0)
        
        # Count edge cases tests
        edge_results = comprehensive_results["test_results"].get("edge_cases", {})
        if "edge_tests" in edge_results:
            edge_test_count = len(edge_results["edge_tests"])
            edge_passed = sum(1 for test in edge_results["edge_tests"] if test.get("success", False))
            total_tests += edge_test_count
            passed_tests += edge_passed
        
        comprehensive_results["overall_summary"] = {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": total_tests - passed_tests,
            "success_rate": round((passed_tests / total_tests * 100) if total_tests > 0 else 0, 1)
        }
        
        logger.info(f"Comprehensive test completed. Overall success rate: {comprehensive_results['overall_summary']['success_rate']}%")
        
        return comprehensive_results
    
    def save_results(self, results: Dict[str, Any], filename: str = None) -> Path:
        """
        Save test results to JSON file.
        
        Args:
            results: Test results dictionary
            filename: Optional filename (default: auto-generated)
            
        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            filename = f"cerebrium_test_results_{timestamp}.json"
        
        filepath = Path(filename)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Test results saved to: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Failed to save results: {str(e)}")
            raise


def main():
    """Main function to handle command line arguments and run tests."""
    parser = argparse.ArgumentParser(description="Test deployed Cerebrium model")
    
    parser.add_argument(
        "--image", 
        type=str, 
        help="Path to image file for single prediction"
    )
    
    parser.add_argument(
        "--api-key", 
        type=str, 
        default=API_KEY,
        help="Cerebrium API key"
    )
    
    parser.add_argument(
        "--model-name", 
        type=str, 
        default="resnet18-classifier",
        help="Name of the deployed model"
    )
    
    parser.add_argument(
        "--test-preset", 
        action="store_true",
        help="Run preset custom tests with known images"
    )
    
    parser.add_argument(
        "--health-check", 
        action="store_true",
        help="Run health check tests"
    )
    
    parser.add_argument(
        "--edge-cases", 
        action="store_true",
        help="Run edge case tests"
    )
    
    parser.add_argument(
        "--comprehensive", 
        action="store_true",
        help="Run all tests (comprehensive test suite)"
    )
    
    parser.add_argument(
        "--top-k", 
        type=int, 
        default=5,
        help="Number of top predictions to return (default: 5)"
    )
    
    parser.add_argument(
        "--save-results", 
        action="store_true",
        help="Save test results to JSON file"
    )
    
    parser.add_argument(
        "--output-file", 
        type=str,
        help="Output filename for saving results"
    )
    
    args = parser.parse_args()
    
    # Initialize tester
    try:
        tester = CerebriumModelTester(
            api_key=args.api_key,
            model_name=args.model_name
        )
    except Exception as e:
        logger.error(f"Failed to initialize tester: {str(e)}")
        return 1
    
    results = None
    
    try:
        if args.image:
            # Single image prediction
            image_path = Path(args.image)
            if not image_path.exists():
                logger.error(f"Image file not found: {image_path}")
                return 1
            
            logger.info(f"Predicting single image: {image_path}")
            result = tester.predict_image(image_path, top_k=args.top_k)
            
            if result.get("success", False):
                print(f"\n🎯 Prediction Results for {image_path.name}:")
                print(f"   Predicted Class ID: {result.get('class_id', 'N/A')}")
                print(f"   Confidence: {result.get('confidence', 'N/A'):.4f}")
                print(f"   Response Time: {result.get('request_time', 'N/A')}s")
                
                if "top_predictions" in result:
                    print(f"\n📊 Top {args.top_k} Predictions:")
                    for i, pred in enumerate(result["top_predictions"], 1):
                        print(f"   {i}. Class {pred['class_id']}: {pred['confidence']:.4f}")
            else:
                print(f"\n❌ Prediction failed: {result.get('message', 'Unknown error')}")
                return 1
        
        elif args.test_preset:
            # Run preset tests with known images
            results = tester.test_known_images()
            
        elif args.health_check:
            # Run health check tests
            results = tester.test_model_health()
            
        elif args.edge_cases:
            # Run edge case tests
            results = tester.test_edge_cases()
            
        elif args.comprehensive:
            # Run comprehensive test suite
            results = tester.run_comprehensive_test()
            
        else:
            # Default: run known images test
            logger.info("No specific test specified. Running preset tests with known images...")
            results = tester.test_known_images()
        
        # Save results if requested
        if results and args.save_results:
            tester.save_results(results, args.output_file)
        
        # Print summary if results available
        if results:
            print(f"\n📈 Test Summary:")
            if "summary" in results:
                summary = results["summary"]
                print(f"   Total Tests: {summary.get('total', 0)}")
                print(f"   Passed: {summary.get('passed', 0)}")
                print(f"   Failed: {summary.get('failed', 0)}")
                print(f"   Success Rate: {summary.get('success_rate', 0)}%")
            elif "overall_summary" in results:
                summary = results["overall_summary"]
                print(f"   Total Tests: {summary.get('total_tests', 0)}")
                print(f"   Passed: {summary.get('passed_tests', 0)}")
                print(f"   Failed: {summary.get('failed_tests', 0)}")
                print(f"   Success Rate: {summary.get('success_rate', 0)}%")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Test execution failed: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(main())