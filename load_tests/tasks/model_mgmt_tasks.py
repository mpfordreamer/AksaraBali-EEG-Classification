"""
Model management workflow tasks for load testing
"""
import logging
from locust import task, TaskSet
from config import API_BASE_URL, API_TIMEOUT, VALIDATE_RESPONSES
from utils.data_helpers import validate_models_list_response, log_response_info

logger = logging.getLogger(__name__)


class ModelManagementTasks(TaskSet):
    """
    Tasks for model management operations.
    
    This simulates users listing available models, downloading models,
    and potentially deleting test models (with safeguards).
    """
    
    @task(5)
    def list_models(self):
        """
        List all available trained models.
        
        Endpoint: GET /models/list
        Expected response: JSON with models array, default_model, etc.
        """
        with self.client.get(
            "/models/list",
            timeout=API_TIMEOUT,
            catch_response=True,
            name="GET /models/list"
        ) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    
                    # Validate response structure
                    if VALIDATE_RESPONSES:
                        if validate_models_list_response(data):
                            log_response_info(data, validate_models_list_response)
                            
                            # Log model count
                            models = data.get('models', [])
                            logger.info(f"Found {len(models)} available models")
                            response.success()
                        else:
                            response.failure("Response validation failed")
                    else:
                        response.success()
                except ValueError as e:
                    response.failure(f"Invalid JSON response: {str(e)}")
            else:
                response.failure(f"HTTP {response.status_code}: {response.text}")
    
    @task(1)
    def delete_non_default_model(self):
        """
        Delete a non-default test model.
        
        Endpoint: DELETE /models/delete/{model_name}
        
        Note: This is a destructive operation. In load testing, we:
        1. First list models to find deletable ones
        2. Only delete models that appear to be test/temporary models
        3. Never delete the default model
        
        For thesis load testing, you might want to disable this task
        or only run it in a controlled test environment.
        """
        # First, get list of models
        try:
            list_response = self.client.get(
                "/models/list",
                timeout=API_TIMEOUT
            )
            
            if list_response.status_code != 200:
                logger.warning("Could not list models for deletion task")
                return
            
            data = list_response.json()
            models = data.get('models', [])
            default_model = data.get('default_model', '')
            
            # Find a deletable test model (exclude default)
            deletable_models = [
                m for m in models 
                if m.get('name', '') != default_model 
                and '_Test_' in m.get('name', '')  # Only delete models with "_Test_" in name
            ]
            
            if not deletable_models:
                logger.debug("No deletable test models found")
                return
            
            # Delete the first deletable model
            model_to_delete = deletable_models[0]['name']
            
            with self.client.delete(
                f"/models/delete/{model_to_delete}",
                timeout=API_TIMEOUT,
                catch_response=True,
                name="DELETE /models/delete/{name}"
            ) as response:
                if response.status_code == 200:
                    logger.info(f"Deleted test model: {model_to_delete}")
                    response.success()
                elif response.status_code == 400:
                    # Trying to delete default or invalid model
                    logger.debug("Cannot delete model (protected or invalid)")
                    response.success()  # Don't penalize in load testing
                elif response.status_code == 404:
                    # Model not found
                    logger.debug(f"Model not found: {model_to_delete}")
                    response.success()
                else:
                    response.failure(f"HTTP {response.status_code}: {response.text}")
                    
        except Exception as e:
            logger.error(f"Error in delete_non_default_model: {str(e)}")
