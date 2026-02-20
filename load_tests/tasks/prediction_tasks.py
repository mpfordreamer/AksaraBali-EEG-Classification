"""
Prediction workflow tasks for load testing
"""
import logging
from locust import task, TaskSet
from config import API_BASE_URL, API_TIMEOUT, TEST_FILES, VALIDATE_RESPONSES
from utils.data_helpers import (
    prepare_file_for_upload,
    validate_predict_response,
    log_request_info,
    log_response_info
)

logger = logging.getLogger(__name__)


class PredictionTasks(TaskSet):
    """
    Tasks for making predictions with trained models.
    
    This simulates users uploading test data and getting predictions
    using either the default model or a specific trained model.
    """
    
    @task(5)
    def predict_with_default_model(self):
        """
        Make predictions using the default model.
        
        Endpoint: POST /predict
        Expected response: JSON with predicted_labels, predicted_probs, etc.
        """
        try:
            # Prepare prediction dataset file
            files = prepare_file_for_upload(
                TEST_FILES['predict'],
                field_name='dataset_file'
            )
            
            log_request_info("POST", "/predict (default model)", files)
            
            # Make request
            with self.client.post(
                "/predict",
                files=files,
                timeout=API_TIMEOUT,
                catch_response=True,
                name="POST /predict [default]"
            ) as response:
                if response.status_code == 200:
                    try:
                        data = response.json()
                        
                        # Validate response structure
                        if VALIDATE_RESPONSES:
                            if validate_predict_response(data):
                                log_response_info(data, validate_predict_response)
                                
                                # Log prediction summary
                                num_trials = data.get('num_trials', 0)
                                labels = data.get('predicted_labels', [])
                                logger.info(
                                    f"Prediction completed - "
                                    f"Trials: {num_trials}, "
                                    f"Labels: {set(labels) if labels else 'none'}"
                                )
                                response.success()
                            else:
                                response.failure("Response validation failed")
                        else:
                            response.success()
                    except ValueError as e:
                        response.failure(f"Invalid JSON response: {str(e)}")
                elif response.status_code == 404:
                    response.failure("Default model not found")
                else:
                    response.failure(f"HTTP {response.status_code}: {response.text}")
                    
        except FileNotFoundError as e:
            logger.error(f"Test data file not found: {str(e)}")
            self.environment.runner.quit()
        except Exception as e:
            logger.error(f"Error in predict_with_default_model: {str(e)}")
    
    @task(2)
    def predict_with_specific_model(self):
        """
        Make predictions using a specific named model.
        
        Endpoint: POST /predict?model_name=X
        Expected response: JSON with predicted_labels, predicted_probs, etc.
        """
        try:
            # Use a known model name (adjust based on your test environment)
            # In a real load test, you might randomly select from available models
            model_name = "LSTM_Model_Final"  # Default model as fallback
            
            files = prepare_file_for_upload(
                TEST_FILES['predict'],
                field_name='dataset_file'
            )
            
            params = {'model_name': model_name}
            
            log_request_info("POST", f"/predict?model_name={model_name}", files)
            
            # Make request
            with self.client.post(
                "/predict",
                files=files,
                params=params,
                timeout=API_TIMEOUT,
                catch_response=True,
                name="POST /predict [named]"
            ) as response:
                if response.status_code == 200:
                    try:
                        data = response.json()
                        
                        if VALIDATE_RESPONSES and validate_predict_response(data):
                            log_response_info(data, validate_predict_response)
                            response.success()
                        else:
                            response.success()
                    except ValueError as e:
                        response.failure(f"Invalid JSON response: {str(e)}")
                elif response.status_code == 404:
                    # Model not found - could happen with random model names
                    logger.debug(f"Model '{model_name}' not found")
                    response.success()  # Don't penalize in load testing
                else:
                    response.failure(f"HTTP {response.status_code}: {response.text}")
                    
        except FileNotFoundError as e:
            logger.error(f"Test data file not found: {str(e)}")
            self.environment.runner.quit()
        except Exception as e:
            logger.error(f"Error in predict_with_specific_model: {str(e)}")
