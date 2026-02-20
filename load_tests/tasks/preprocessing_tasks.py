"""
Preprocessing workflow tasks for load testing
"""
import os
import logging
from locust import task, TaskSet
from config import API_BASE_URL, API_TIMEOUT, TEST_FILES, VALIDATE_RESPONSES
from utils.data_helpers import (
    prepare_multipart_files,
    validate_preprocess_response,
    log_request_info,
    log_response_info
)

logger = logging.getLogger(__name__)


class PreprocessingTasks(TaskSet):
    """
    Tasks for EEG data preprocessing workflow.
    
    This simulates a user uploading baseline and training EEG data,
    running preprocessing, and downloading the processed features.
    """
    
    @task(3)
    def preprocess_eeg_data(self):
        """
        Upload baseline and training files for preprocessing.
        
        Endpoint: POST /preprocess
        Expected response: JSON with trial_features_shape, base_features_shape, etc.
        """
        try:
            # Prepare files for upload
            files = prepare_multipart_files({
                'baseline_file': TEST_FILES['baseline'],
                'training_file': TEST_FILES['dataset_awal']
            })
            
            log_request_info("POST", "/preprocess", files)
            
            # Make request
            with self.client.post(
                "/preprocess",
                files=files,
                timeout=API_TIMEOUT,
                catch_response=True,
                name="POST /preprocess"
            ) as response:
                if response.status_code == 200:
                    try:
                        data = response.json()
                        
                        # Validate response structure
                        if VALIDATE_RESPONSES:
                            if validate_preprocess_response(data):
                                log_response_info(data, validate_preprocess_response)
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
            self.environment.runner.quit()  # Stop test if data files are missing
        except Exception as e:
            logger.error(f"Error in preprocess_eeg_data: {str(e)}")
    
    @task(1)
    def download_preprocessed_data(self):
        """
        Download preprocessed data files.
        
        Endpoint: POST /preprocess/download
        Expected response: Binary .mat file
        
        Note: This should be called after preprocess_eeg_data, but in load testing
        we simulate independent requests as users may download at different times.
        """
        with self.client.post(
            "/preprocess/download",
            timeout=API_TIMEOUT,
            catch_response=True,
            name="POST /preprocess/download"
        ) as response:
            if response.status_code == 200:
                # Check if response is binary data
                content_type = response.headers.get('Content-Type', '')
                if 'octet-stream' in content_type or response.content:
                    # Successful binary download
                    file_size_mb = len(response.content) / (1024 * 1024)
                    logger.info(f"Downloaded preprocessed data: {file_size_mb:.2f} MB")
                    response.success()
                else:
                    response.failure("Response is not binary data")
            elif response.status_code == 400:
                # Expected if no recent preprocessing - not a failure in load testing context
                logger.debug("No preprocessed data available (expected in parallel load testing)")
                response.success()  # Mark as success to avoid skewing metrics
            else:
                response.failure(f"HTTP {response.status_code}: {response.text}")
