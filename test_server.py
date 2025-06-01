#!/usr/bin/env python3
"""
Enhanced test script for deployed Cerebrium model with debugging capabilities.
Tests the deployed model and provides monitoring capabilities with better error handling.
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
    Enhanced test client for deployed Cerebrium model with debugging capabilities.
    """
    
    def __init__(self, api_key: str, project_name: str, model_name: str = "resnet18-classifier"):
        """
        Initialize the tester.
        
        Args:
            api_key: Cerebrium API key
            project_name: Name of the Cerebrium project
            model_name: Name of the deployed model
        """
        self.api_key = api_key
        self.project_name = project_name
        self.model_name = model_name
        self.base_url = f"{CEREBRIUM_API_BASE}/p/{project_name}/predict"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        logger.info(f"Initialized tester for project: {project_name}, model: {model_name}")
        logger.info(f"API endpoint: {self.base_url}")
    
    def debug_api_connection(self) -> Dict[str, Any]:
        """
        Debug API connection and endpoint availability.
        
        Returns:
            Debug information about the API connection
        """
        logger.info("Starting API connection debugging...")
        
        debug_info = {
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "api_key_status": "present" if self.api_key else "missing",
            "project_name": self.project_name,
            "model_name": self.model_name,
            "endpoint_url": self.base_url,
            "tests": []
        }
        
        # Test 1: Check if API key is valid format
        api_key_test = {
            "test": "API Key Format",
            "status": "pass" if len(self.api_key) > 10 else "fail",
            "details": f"API key length: {len(self.api_key)}"
        }
        debug_info["tests"].append(api_key_test)
        
        # Test 2: Try different endpoint variations
        endpoint_variations = [
            f"{CEREBRIUM_API_BASE}/p/{self.project_name}/predict",
            f"{CEREBRIUM_API_BASE}/projects/{self.project_name}/predict",
            f"{CEREBRIUM_API_BASE}/{self.project_name}/predict",
            f"https://api.cerebrium.ai/v4/p/{self.project_name}/predict",
            f"https://api.cerebrium.ai/v3/p/{self.project_name}/predict",
            # Also try with model name in path
            f"{CEREBRIUM_API_BASE}/p/{self.project_name}/{self.model_name}/predict",
            f"{CEREBRIUM_API_BASE}/projects/{self.project_name}/{self.model_name}/predict"
        ]
        
        for endpoint in endpoint_variations:
            endpoint_test = {
                "test": f"Endpoint Test",
                "endpoint": endpoint,
                "status": "unknown",
                "details": ""
            }
            
            try:
                # Try a simple GET request first to see if endpoint exists
                response = requests.get(
                    endpoint,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    timeout=10
                )
                
                endpoint_test["status_code"] = response.status_code
                endpoint_test["response_headers"] = dict(response.headers)
                
                if response.status_code == 404:
                    endpoint_test["status"] = "not_found"
                    endpoint_test["details"] = "Endpoint not found (404)"
                elif response.status_code == 405:
                    endpoint_test["status"] = "method_not_allowed"
                    endpoint_test["details"] = "Method not allowed (405) - endpoint exists but GET not supported"
                elif response.status_code == 401:
                    endpoint_test["status"] = "unauthorized"
                    endpoint_test["details"] = "Unauthorized (401) - check API key"
                elif response.status_code == 403:
                    endpoint_test["status"] = "forbidden"
                    endpoint_test["details"] = "Forbidden (403) - API key may be invalid"
                else:
                    endpoint_test["status"] = "accessible"
                    endpoint_test["details"] = f"HTTP {response.status_code}"
                
                # Try to get response content
                try:
                    endpoint_test["response_body"] = response.text[:500]  # First 500 chars
                except:
                    pass
                    
            except requests.exceptions.Timeout:
                endpoint_test["status"] = "timeout"
                endpoint_test["details"] = "Request timed out"
            except requests.exceptions.ConnectionError:
                endpoint_test["status"] = "connection_error"
                endpoint_test["details"] = "Connection failed"
            except Exception as e:
                endpoint_test["status"] = "error"
                endpoint_test["details"] = str(e)
            
            debug_info["tests"].append(endpoint_test)
            logger.info(f"Tested endpoint {endpoint}: {endpoint_test['status']}")
            
            # If we found a working endpoint, update the base URL
            if endpoint_test["status"] in ["accessible", "method_not_allowed"]:
                logger.info(f"Found working endpoint: {endpoint}")
                if endpoint != self.base_url:
                    logger.info(f"Updating base URL from {self.base_url} to {endpoint}")
                    self.base_url = endpoint
                break
        
        # Test 3: List available projects (if API supports it)
        list_projects_test = {
            "test": "List Projects",
            "status": "unknown",
            "details": ""
        }
        
        try:
            list_url = f"{CEREBRIUM_API_BASE}/projects"
            response = requests.get(
                list_url,
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=10
            )
            
            list_projects_test["status_code"] = response.status_code
            if response.status_code == 200:
                try:
                    projects = response.json()
                    list_projects_test["status"] = "success"
                    list_projects_test["details"] = f"Found {len(projects)} projects"
                    list_projects_test["projects"] = projects
                    
                    # Check if our project exists
                    if isinstance(projects, list):
                        project_names = [p.get('name', '') for p in projects if isinstance(p, dict)]
                        if self.project_name in project_names:
                            list_projects_test["project_found"] = True
                            list_projects_test["details"] += f" - Project '{self.project_name}' found"
                        else:
                            list_projects_test["project_found"] = False
                            list_projects_test["details"] += f" - Project '{self.project_name}' NOT found"
                            list_projects_test["available_projects"] = project_names
                            
                except:
                    list_projects_test["status"] = "success"
                    list_projects_test["details"] = "Response received but couldn't parse JSON"
            else:
                list_projects_test["status"] = "failed"
                list_projects_test["details"] = f"HTTP {response.status_code}"
                
        except Exception as e:
            list_projects_test["status"] = "error"
            list_projects_test["details"] = str(e)
        
        debug_info["tests"].append(list_projects_test)
        
        return debug_info
    
    def test_minimal_request(self) -> Dict[str, Any]:
        """
        Test with minimal request to isolate issues.
        
        Returns:
            Test results
        """
        logger.info("Testing minimal request...")
        
        # Create the smallest possible valid image
        tiny_image = np.ones((1, 1, 3), dtype=np.uint8) * 128  # Gray pixel
        pil_image = Image.fromarray(tiny_image)
        
        # Convert to base64
        buffer = io.BytesIO()
        pil_image.save(buffer, format='JPEG')
        image_bytes = buffer.getvalue()
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        
        # Test different request variations
        test_requests = [
            {
                "name": "Minimal request",
                "data": {"image": image_base64}
            },
            {
                "name": "With top_k",
                "data": {"image": image_base64, "top_k": 1}
            },
            {
                "name": "With top_k=5",
                "data": {"image": image_base64, "top_k": 5}
            }
        ]
        
        results = []
        
        for test_request in test_requests:
            logger.info(f"Testing: {test_request['name']}")
            
            try:
                start_time = time.time()
                
                # Make the request
                response = requests.post(
                    self.base_url,
                    headers=self.headers,
                    json=test_request["data"],
                    timeout=30
                )
                
                request_time = time.time() - start_time
                
                result = {
                    "test_name": test_request["name"],
                    "status_code": response.status_code,
                    "request_time": round(request_time, 4),
                    "success": response.status_code == 200
                }
                
                # Try to parse response
                try:
                    response_data = response.json()
                    result["response"] = response_data
                except:
                    result["response_text"] = response.text[:1000]  # First 1000 chars
                
                # Add headers for debugging
                result["response_headers"] = dict(response.headers)
                
                results.append(result)
                
                if response.status_code == 200:
                    logger.info(f"✅ {test_request['name']}: Success")
                else:
                    logger.warning(f"❌ {test_request['name']}: HTTP {response.status_code}")
                    
            except Exception as e:
                result = {
                    "test_name": test_request["name"],
                    "error": str(e),
                    "success": False
                }
                results.append(result)
                logger.error(f"❌ {test_request['name']}: Exception - {str(e)}")
        
        return {
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "project_name": self.project_name,
            "model_name": self.model_name,
            "minimal_tests": results
        }
    
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
                
                # Resize if too large (common issue)
                if img.size[0] > 1024 or img.size[1] > 1024:
                    img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
                    logger.info(f"Resized image to {img.size}")
                
                # Convert to bytes
                buffer = io.BytesIO()
                img.save(buffer, format='JPEG', quality=85)
                image_bytes = buffer.getvalue()
                
                # Log image info
                logger.info(f"Image size: {img.size}, Mode: {img.mode}, Bytes: {len(image_bytes)}")
                
                # Encode to base64
                return base64.b64encode(image_bytes).decode('utf-8')
                
        except Exception as e:
            raise ValueError(f"Failed to encode image {image_path}: {str(e)}")
    
    def predict_image(self, image_path: Path, top_k: int = 5) -> Dict[str, Any]:
        """
        Send prediction request to deployed model with enhanced error handling.
        
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
            logger.info(f"Base64 encoded image length: {len(image_base64)}")
            
            # Prepare request
            request_data = {
                "image": image_base64,
                "top_k": top_k
            }
            
            logger.info(f"Making request to: {self.base_url}")
            logger.info(f"Request data keys: {list(request_data.keys())}")
            logger.info(f"Headers: {self.headers}")
            
            # Send request
            start_time = time.time()
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json=request_data,
                timeout=60  # Increased timeout
            )
            request_time = time.time() - start_time
            
            logger.info(f"Response status code: {response.status_code}")
            logger.info(f"Response headers: {dict(response.headers)}")
            
            # Log response content for debugging
            response_text = response.text
            logger.info(f"Response text (first 500 chars): {response_text[:500]}")
            
            # Check response
            if response.status_code != 200:
                return {
                    "success": False,
                    "message": f"HTTP {response.status_code}: {response_text}",
                    "request_time": request_time,
                    "status_code": response.status_code,
                    "response_headers": dict(response.headers)
                }
            
            # Parse JSON response
            try:
                result = response.json()
            except json.JSONDecodeError as e:
                return {
                    "success": False,
                    "message": f"Invalid JSON response: {str(e)}",
                    "request_time": request_time,
                    "response_text": response_text[:1000]
                }
            
            # Add timing information
            result['request_time'] = round(request_time, 4)
            
            logger.info(f"Prediction completed in {request_time:.4f}s")
            
            return result
            
        except requests.exceptions.Timeout:
            logger.error("Request timed out")
            return {
                "success": False,
                "message": "Request timed out after 60 seconds",
                "request_time": 60
            }
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Connection error: {str(e)}")
            return {
                "success": False,
                "message": f"Connection error: {str(e)}",
                "request_time": 0
            }
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
    
    def comprehensive_debug(self, test_image_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Run comprehensive debugging to identify the issue.
        
        Args:
            test_image_path: Optional path to test image
            
        Returns:
            Complete debug results
        """
        logger.info("Starting comprehensive debugging...")
        
        debug_results = {
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "debug_suite": "comprehensive",
            "project_name": self.project_name,
            "model_name": self.model_name,
            "results": {}
        }
        
        # 1. API Connection Debug
        logger.info("=" * 50)
        logger.info("DEBUGGING API CONNECTION")
        logger.info("=" * 50)
        debug_results["results"]["api_debug"] = self.debug_api_connection()
        
        # 2. Minimal Request Test
        logger.info("=" * 50)
        logger.info("TESTING MINIMAL REQUEST")
        logger.info("=" * 50)
        debug_results["results"]["minimal_test"] = self.test_minimal_request()
        
        # 3. Image Test (if image provided)
        if test_image_path and test_image_path.exists():
            logger.info("=" * 50)
            logger.info("TESTING WITH PROVIDED IMAGE")
            logger.info("=" * 50)
            debug_results["results"]["image_test"] = self.predict_image(test_image_path)
        
        # 4. Generate recommendations
        recommendations = []
        
        # Check API debug results
        api_debug = debug_results["results"]["api_debug"]
        working_endpoints = [test for test in api_debug["tests"] 
                           if test.get("status") in ["accessible", "method_not_allowed"]]
        
        if not working_endpoints:
            recommendations.append(f"No working endpoints found. Check if the project '{self.project_name}' is deployed and the project name is correct.")
        
        if api_debug["api_key_status"] == "missing":
            recommendations.append("API key is missing or too short. Verify your Cerebrium API key.")
        
        # Check if project was found in listing
        project_list_test = next((test for test in api_debug["tests"] if test.get("test") == "List Projects"), None)
        if project_list_test and project_list_test.get("project_found") == False:
            available_projects = project_list_test.get("available_projects", [])
            recommendations.append(f"Project '{self.project_name}' not found. Available projects: {available_projects}")
        
        # Check minimal test results
        minimal_test = debug_results["results"]["minimal_test"]
        successful_tests = [test for test in minimal_test["minimal_tests"] if test.get("success")]
        
        if not successful_tests:
            recommendations.append("All minimal tests failed. This suggests a fundamental API or authentication issue.")
        
        # Check for common error patterns
        for test in minimal_test["minimal_tests"]:
            if test.get("status_code") == 404:
                recommendations.append(f"404 errors suggest the project endpoint doesn't exist. Double-check the project name '{self.project_name}' and deployment status.")
            elif test.get("status_code") == 401:
                recommendations.append("401 errors suggest authentication issues. Verify your API key is correct and has proper permissions.")
            elif test.get("status_code") == 403:
                recommendations.append("403 errors suggest the API key is invalid or doesn't have access to this project.")
        
        debug_results["recommendations"] = recommendations
        
        logger.info("=" * 50)
        logger.info("DEBUGGING COMPLETED")
        logger.info("=" * 50)
        
        if recommendations:
            logger.info("RECOMMENDATIONS:")
            for i, rec in enumerate(recommendations, 1):
                logger.info(f"{i}. {rec}")
        
        return debug_results


def main():
    """Main entry point for the enhanced test script."""
    parser = argparse.ArgumentParser(description="Enhanced Cerebrium model tester with debugging")
    parser.add_argument("--image", type=str, help="Path to test image")
    parser.add_argument("--api-key", type=str, default=API_KEY, help="Cerebrium API key")
    parser.add_argument("--project-name", type=str, required=True, help="Cerebrium project name")
    parser.add_argument("--model-name", type=str, default="resnet18-classifier", help="Deployed model name")
    parser.add_argument("--test-type", type=str, choices=[
        "debug", "minimal", "single", "api-debug"
    ], default="debug", help="Type of test to run")
    parser.add_argument("--output", type=str, default="debug_results.json", help="Output file for results")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize tester
    try:
        tester = CerebriumModelTester(args.api_key, args.project_name, args.model_name)
    except Exception as e:
        logger.error(f"Failed to initialize tester: {str(e)}")
        return 1
    
    # Run tests based on type
    try:
        if args.test_type == "debug":
            image_path = Path(args.image) if args.image else None
            results = tester.comprehensive_debug(image_path)
            
        elif args.test_type == "api-debug":
            results = tester.debug_api_connection()
            
        elif args.test_type == "minimal":
            results = tester.test_minimal_request()
            
        elif args.test_type == "single":
            if not args.image:
                logger.error("--image is required for single image testing")
                return 1
            
            image_path = Path(args.image)
            if not image_path.exists():
                logger.error(f"Image not found: {image_path}")
                return 1
            
            result = tester.predict_image(image_path)
            print(json.dumps(result, indent=2))
            return 0 if result.get("success", False) else 1
        
        # Save results
        try:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to: {args.output}")
        except Exception as e:
            logger.error(f"Failed to save results: {str(e)}")
        
        # Print summary
        print("\n" + "="*60)
        print("DEBUG SUMMARY")
        print("="*60)
        print(json.dumps(results, indent=2))
        
        return 0
            
    except KeyboardInterrupt:
        logger.info("Testing interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Testing failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())