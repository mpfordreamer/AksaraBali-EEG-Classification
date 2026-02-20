"""
Training workflow tasks for load testing
"""
import logging
from locust import task, TaskSet
from config import API_BASE_URL, API_TIMEOUT, TEST_FILES, VALIDATE_RESPONSES
from utils.data_helpers import (
    prepare_file_for_upload,
    validate_train_response,
    log_request_info,
    log_response_info
)

logger = logging.getLogger(__name__)


class TrainingTasks(TaskSet):
    """
    Tasks for model training workflow.
    
    This simulates a user uploading preprocessed data, training a model,
    saving it, and downloading the trained model file.
    """
    
    def __init__(self, parent):
        super().__init__(parent)
        self.last_model_path = None  # Track trained model for download
    
    @task(5)
    def train_model(self):
        """
        Train an LSTM model with preprocessed data.
        
        Endpoint: POST /train
        Expected response: JSON with training metrics (accuracy, loss, f1, etc.)
        
        Note: This is a heavy operation that can take 30-180 seconds.
        """
        try:
            # Prepare preprocessed dataset file
            files = prepare_file_for_upload(
                TEST_FILES['preprocessed'],
                field_name='dataset_file'
            )
            
            log_request_info("POST", "/train", files)
            
            # Make request with extended timeout for training
            with self.client.post(
                "/train",
                files=files,
                timeout=API_TIMEOUT,
                catch_response=True,
                name="POST /train"
            ) as response:
                if response.status_code == 200:
                    try:
                        data = response.json()
                        
                        # Store model path for potential download
                        if 'model_path' in data:
                            self.last_model_path = data['model_path']
                        
                        # Validate response structure
                        if VALIDATE_RESPONSES:
                            if validate_train_response(data):
                                log_response_info(data, validate_train_response)
                                
                                # Log key metrics
                                logger.info(
                                    f"Training completed - "
                                    f"Accuracy: {data.get('val_accuracy', 0):.4f}, "
                                    f"F1: {data.get('f1', 0):.4f}, "
                                    f"Loss: {data.get('val_loss', 0):.4f}"
                                )
                                response.success()
                            else:
                                response.failure("Response validation failed")
                        else:
                            response.success()
                    except ValueError as e:
                        response.failure(f"Invalid JSON response: {str(e)}")
                else:
                    response.failure(f"HTTP {response.status_code}: {response.text}")
                    
        except FileNotFoundError as e:
            logger.error(f"Test data file not found: {str(e)}")
            self.environment.runner.quit()
        except Exception as e:
            logger.error(f"Error in train_model: {str(e)}")
    
    @task(0)
    def save_trained_model(self):
        """
        Save the recently trained model to disk.
        
        Endpoint: POST /train/save
        Expected response: JSON with model_path and success message
        
        Note: Should be called after train_model, but in load testing
        we simulate independent requests.
        """
        with self.client.post(
            "/train/save",
            timeout=API_TIMEOUT,
            catch_response=True,
            name="POST /train/save"
        ) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if 'model_path' in data:
                        logger.info(f"Model saved: {data['model_path']}")
                        response.success()
                    else:
                        response.failure("model_path not in response")
                except ValueError as e:
                    response.failure(f"Invalid JSON response: {str(e)}")
            elif response.status_code == 400:
                # Expected if no recent training - not a failure in load testing
                logger.debug("No trained model to save (expected in parallel load testing)")
                response.success()  # Mark as success to avoid skewing metrics
            else:
                response.failure(f"HTTP {response.status_code}: {response.text}")
    
    @task(1)
    def download_model(self):
        """
        Download a trained model file.
        
        Endpoint: GET /train/download
        Expected response: Binary .h5 model file
        """
        # Try to download the last trained model, or default model
        params = {}
        if self.last_model_path:
            # Extract just the filename
            import os
            model_name = os.path.basename(self.last_model_path)
            params = {'model_name': model_name}
        
        with self.client.get(
            "/train/download",
            params=params,
            timeout=API_TIMEOUT,
            catch_response=True,
            name="GET /train/download"
        ) as response:
            if response.status_code == 200:
                # Check if response is binary data
                content_type = response.headers.get('Content-Type', '')
                if 'octet-stream' in content_type or response.content:
                    file_size_mb = len(response.content) / (1024 * 1024)
                    logger.info(f"Downloaded model: {file_size_mb:.2f} MB")
                    response.success()
                else:
                    response.failure("Response is not binary data")
            elif response.status_code == 400 or response.status_code == 404:
                # Model not found - could be expected in some scenarios
                logger.debug("Model not available for download")
                response.success()  # Don't penalize for this in load testing
            else:
                response.failure(f"HTTP {response.status_code}: {response.text}")
    
    @task(1)
    def get_confusion_matrix(self):
        """
        Get the confusion matrix visualization from training.
        
        Endpoint: GET /train/plot/confusion-matrix
        Expected response: PNG image
        """
        with self.client.get(
            "/train/plot/confusion-matrix",
            timeout=API_TIMEOUT,
            catch_response=True,
            name="GET /train/plot/confusion-matrix"
        ) as response:
            if response.status_code == 200:
                content_type = response.headers.get('Content-Type', '')
                if 'image/png' in content_type:
                    image_size_kb = len(response.content) / 1024
                    logger.info(f"Retrieved confusion matrix: {image_size_kb:.2f} KB")
                    response.success()
                else:
                    response.failure(f"Unexpected content type: {content_type}")
            elif response.status_code == 400:
                # No training results yet - expected in some scenarios
                logger.debug("No confusion matrix available (no recent training)")
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}: {response.text}")
