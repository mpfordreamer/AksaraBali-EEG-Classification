"""
Locust Load Test for Aksara Bali EEG Classification API

This file defines various user behaviors for load testing the API.
Each user class simulates different usage patterns and workflows.

Usage:
    # Web UI mode (recommended for visualization)
    locust --locustfile locustfile.py --host http://localhost:8001
    
    # Headless mode with HTML and CSV reports
    locust --locustfile locustfile.py --headless --users 10 --spawn-rate 2 \\
        --run-time 60s --host http://localhost:8001 \\
        --html reports/load_test_report.html \\
        --csv reports/load_test_stats
    
    # Test specific user type only
    locust --locustfile locustfile.py --headless --users 5 --spawn-rate 1 \\
        --run-time 30s --host http://localhost:8001 --user HealthCheckUser

For more information, see README.md
"""
import logging
from locust import HttpUser, between, task
from config import MIN_WAIT, MAX_WAIT, TRAIN_MIN_WAIT, TRAIN_MAX_WAIT
from tasks.preprocessing_tasks import PreprocessingTasks
from tasks.training_tasks import TrainingTasks
from tasks.prediction_tasks import PredictionTasks
from tasks.model_mgmt_tasks import ModelManagementTasks

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============ USER CLASSES ============

class HealthCheckUser(HttpUser):
    """
    Lightweight user that only performs health checks.
    
    Use case: Monitoring systems, load balancers, uptime checkers
    Weight: High frequency (2x)
    """
    weight = 2
    wait_time = between(MIN_WAIT / 1000, MAX_WAIT / 1000)
    
    @task
    def health_check(self):
        """Check API health endpoint."""
        with self.client.get(
            "/health",
            catch_response=True,
            name="GET /health"
        ) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if data.get('status') == 'ok':
                        response.success()
                    else:
                        response.failure("Health check status not 'ok'")
                except ValueError:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"HTTP {response.status_code}")


class PreprocessingUser(HttpUser):
    """
    User performing EEG data preprocessing workflows.
    
    Use case: Researchers uploading and preprocessing new EEG datasets
    Weight: Low frequency (1x) - preprocessing is done occasionally
    """
    weight = 1
    wait_time = between(MIN_WAIT / 1000, MAX_WAIT / 1000)
    tasks = [PreprocessingTasks]


class TrainingUser(HttpUser):
    """
    User training new models.
    
    Use case: Researchers training models with different hyperparameters
    Weight: Low frequency (1x) - training is computationally expensive
    Wait time: Longer waits to simulate thinking time between operations
    """
    weight = 1
    wait_time = between(TRAIN_MIN_WAIT / 1000, TRAIN_MAX_WAIT / 1000)
    tasks = [TrainingTasks]


class PredictionUser(HttpUser):
    wait_time = between(2, 5)
    """
    User making predictions with trained models.
    
    Use case: Most common operation - using the API for inference
    Weight: High frequency (3x) - prediction is the main use case
    """
    weight = 3
    wait_time = between(MIN_WAIT / 1000, MAX_WAIT / 1000)
    tasks = [PredictionTasks]


class ModelManagementUser(HttpUser):
    """
    User managing models (listing, downloading).
    
    Use case: Researchers browsing available models, downloading for local use
    Weight: Medium frequency (1x)
    """
    weight = 1
    wait_time = between(MIN_WAIT / 1000, MAX_WAIT / 1000)
    tasks = [ModelManagementTasks]


class MixedWorkflowUser(HttpUser):
    """
    User performing complete end-to-end workflows.
    
    Use case: Simulates a real user going through the full pipeline:
              preprocessing → training → prediction → model management
    Weight: Medium frequency (2x)
    """
    weight = 2
    wait_time = between(MIN_WAIT / 1000, MAX_WAIT / 1000)
    
    @task(2)
    def health_check(self):
        """Occasionally check health."""
        with self.client.get("/health", catch_response=True, name="GET /health") as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}")
    
    @task(3)
    def list_and_predict(self):
        """
        Common workflow: List available models, then make a prediction.
        """
        # First, list models
        list_response = self.client.get(
            "/models/list",
            catch_response=True,
            name="GET /models/list"
        )
        
        if list_response.status_code != 200:
            return
        
        # Then make a prediction
        from config import TEST_FILES
        from utils.data_helpers import prepare_file_for_upload
        
        try:
            files = prepare_file_for_upload(
                TEST_FILES['predict'],
                field_name='dataset_file'
            )
            
            with self.client.post(
                "/predict",
                files=files,
                catch_response=True,
                name="POST /predict [workflow]"
            ) as response:
                if response.status_code == 200:
                    response.success()
                else:
                    response.failure(f"HTTP {response.status_code}")
        except Exception as e:
            logger.error(f"Error in list_and_predict: {str(e)}")
    
    @task(1)
    def preprocess_workflow(self):
        """
        Complete preprocessing workflow: preprocess → download.
        """
        from config import TEST_FILES
        from utils.data_helpers import prepare_multipart_files
        
        try:
            # Step 1: Preprocess
            files = prepare_multipart_files({
                'baseline_file': TEST_FILES['baseline'],
                'training_file': TEST_FILES['dataset_awal']
            })
            
            preprocess_response = self.client.post(
                "/preprocess",
                files=files,
                catch_response=True,
                name="POST /preprocess [workflow]"
            )
            
            if preprocess_response.status_code != 200:
                return
            
            # Step 2: Download (if preprocessing succeeded)
            with self.client.post(
                "/preprocess/download",
                catch_response=True,
                name="POST /preprocess/download [workflow]"
            ) as response:
                if response.status_code == 200:
                    response.success()
                elif response.status_code == 400:
                    # Expected in parallel testing
                    response.success()
                else:
                    response.failure(f"HTTP {response.status_code}")
                    
        except Exception as e:
            logger.error(f"Error in preprocess_workflow: {str(e)}")

    # In locustfile.py, for workflow tasks
    @task(1)
    def predict_workflow(self):
        try:
            with self.client.post(
                "/predict",
                files={"dataset_file": ...},
                catch_response=True,
                timeout=60  # Add explicit timeout
            ) as response:
                if response.status_code == 0:
                    response.failure("Connection timeout")
                elif response.status_code >= 400:
                    response.failure(f"HTTP {response.status_code}")
        except Exception as e:
            print(f"Workflow prediction failed: {e}")


# ============ EVENT HANDLERS ============

from locust import events

@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """
    Called when the test starts.
    Perform any necessary setup here.
    """
    logger.info("="*60)
    logger.info("Starting Aksara Bali EEG API Load Test")
    logger.info("="*60)
    
    # Verify test data files exist
    from config import TEST_FILES
    import os
    
    missing_files = []
    for name, path in TEST_FILES.items():
        if not os.path.exists(path):
            missing_files.append(f"{name}: {path}")
    
    if missing_files:
        logger.error("Missing test data files:")
        for f in missing_files:
            logger.error(f"  - {f}")
        logger.error("Please copy test data files to load_tests/test_data/")
        environment.runner.quit()


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """
    Called when the test stops.
    Log summary information.
    """
    logger.info("="*60)
    logger.info("Load Test Completed")
    logger.info("="*60)
    
    # Log summary stats
    stats = environment.stats
    logger.info(f"Total requests: {stats.total.num_requests}")
    logger.info(f"Total failures: {stats.total.num_failures}")
    logger.info(f"Average response time: {stats.total.avg_response_time:.2f}ms")
    logger.info(f"Requests per second: {stats.total.total_rps:.2f}")
    
    if stats.total.num_requests > 0:
        failure_rate = (stats.total.num_failures / stats.total.num_requests) * 100
        logger.info(f"Failure rate: {failure_rate:.2f}%")


@events.request.add_listener
def on_request(request_type, name, response_time, response_length, exception, **kwargs):
    """
    Called after each request.
    Log slow requests or errors.
    """
    # Log very slow requests (> 60 seconds)
    if response_time > 60000:
        logger.warning(
            f"Slow request detected: {request_type} {name} "
            f"took {response_time:.2f}ms"
        )
    
    # Log errors
    if exception:
        logger.error(
            f"Request failed: {request_type} {name} - {str(exception)}"
        )
