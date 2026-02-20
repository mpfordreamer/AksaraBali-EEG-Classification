"""
Helper utilities for handling test data files and responses
"""
import os
import logging
from typing import Dict, Any, Optional
import scipy.io as sio
import numpy as np

logger = logging.getLogger(__name__)


def load_mat_file(file_path: str) -> Dict[str, Any]:
    """
    Load a .mat file and return its contents.
    
    Args:
        file_path: Path to the .mat file
        
    Returns:
        Dictionary containing the .mat file data
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file can't be loaded
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Test data file not found: {file_path}")
    
    try:
        data = sio.loadmat(file_path, squeeze_me=True, struct_as_record=False)
        logger.debug(f"Successfully loaded {file_path}")
        return data
    except Exception as e:
        raise ValueError(f"Error loading .mat file {file_path}: {str(e)}")


def prepare_file_for_upload(file_path: str, field_name: str = "file") -> Dict[str, tuple]:
    """
    Prepare a file for multipart form upload.
    
    Args:
        file_path: Path to the file
        field_name: Form field name for the file
        
    Returns:
        Dictionary ready for requests library 'files' parameter
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    filename = os.path.basename(file_path)
    
    with open(file_path, 'rb') as f:
        file_content = f.read()
    
    # Return format expected by requests library
    # files = {field_name: (filename, file_content, 'application/octet-stream')}
    return {field_name: (filename, file_content, 'application/octet-stream')}


def prepare_multipart_files(file_paths: Dict[str, str]) -> Dict[str, tuple]:
    """
    Prepare multiple files for multipart form upload.
    
    Args:
        file_paths: Dictionary mapping field names to file paths
        
    Returns:
        Dictionary ready for requests library 'files' parameter
    """
    files = {}
    for field_name, file_path in file_paths.items():
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found for field '{field_name}': {file_path}")
        
        filename = os.path.basename(file_path)
        with open(file_path, 'rb') as f:
            file_content = f.read()
        
        files[field_name] = (filename, file_content, 'application/octet-stream')
    
    return files


def validate_response(response_data: Dict[str, Any], expected_keys: list) -> bool:
    """
    Validate that a response contains expected keys.
    
    Args:
        response_data: Response JSON data
        expected_keys: List of keys that should be present
        
    Returns:
        True if all keys are present, False otherwise
    """
    if not isinstance(response_data, dict):
        logger.warning(f"Response is not a dictionary: {type(response_data)}")
        return False
    
    missing_keys = [key for key in expected_keys if key not in response_data]
    
    if missing_keys:
        logger.warning(f"Missing keys in response: {missing_keys}")
        return False
    
    return True


def validate_preprocess_response(response_data: Dict[str, Any]) -> bool:
    """Validate preprocessing endpoint response structure."""
    required_keys = ["message", "trial_features_shape", "base_features_shape"]
    return validate_response(response_data, required_keys)


def validate_train_response(response_data: Dict[str, Any]) -> bool:
    """Validate training endpoint response structure."""
    required_keys = ["val_accuracy", "val_loss", "precision", "recall", "f1", "model_path"]
    
    if not validate_response(response_data, required_keys):
        return False
    
    # Additional validation: check metric ranges
    metrics = ["val_accuracy", "precision", "recall", "f1"]
    for metric in metrics:
        value = response_data.get(metric)
        if not isinstance(value, (int, float)) or not (0 <= value <= 1):
            logger.warning(f"Metric {metric} has invalid value: {value}")
            return False
    
    return True


def validate_predict_response(response_data: Dict[str, Any]) -> bool:
    """Validate prediction endpoint response structure."""
    required_keys = ["message", "predicted_labels", "predicted_probs", "num_trials"]
    
    if not validate_response(response_data, required_keys):
        return False
    
    # Additional validation: check arrays match num_trials
    num_trials = response_data.get("num_trials", 0)
    pred_labels = response_data.get("predicted_labels", [])
    pred_probs = response_data.get("predicted_probs", [])
    
    if len(pred_labels) != num_trials or len(pred_probs) != num_trials:
        logger.warning(
            f"Prediction array length mismatch. "
            f"num_trials={num_trials}, labels={len(pred_labels)}, probs={len(pred_probs)}"
        )
        return False
    
    return True


def validate_models_list_response(response_data: Dict[str, Any]) -> bool:
    """Validate models list endpoint response structure."""
    required_keys = ["models", "default_model"]
    
    if not validate_response(response_data, required_keys):
        return False
    
    # Additional validation: models should be a list
    if not isinstance(response_data.get("models"), list):
        logger.warning("Models field is not a list")
        return False
    
    return True


def get_file_size_mb(file_path: str) -> float:
    """
    Get file size in megabytes.
    
    Args:
        file_path: Path to the file
        
    Returns:
        File size in MB
    """
    if not os.path.exists(file_path):
        return 0.0
    
    size_bytes = os.path.getsize(file_path)
    return size_bytes / (1024 * 1024)


def log_request_info(method: str, url: str, files: Optional[Dict] = None):
    """Log information about an API request."""
    msg = f"{method} {url}"
    
    if files:
        file_info = []
        for field_name, file_data in files.items():
            if isinstance(file_data, tuple) and len(file_data) >= 2:
                filename, content = file_data[0], file_data[1]
                size_mb = len(content) / (1024 * 1024)
                file_info.append(f"{field_name}={filename} ({size_mb:.2f}MB)")
        
        if file_info:
            msg += f" | Files: {', '.join(file_info)}"
    
    logger.info(msg)


def log_response_info(response_data: Dict[str, Any], validation_func=None):
    """Log information about an API response."""
    if validation_func:
        is_valid = validation_func(response_data)
        logger.info(f"Response validation: {'PASSED' if is_valid else 'FAILED'}")
    
    # Log key metrics from response
    if isinstance(response_data, dict):
        metrics = {}
        
        # Common metric keys
        metric_keys = ["val_accuracy", "accuracy", "f1", "precision", "recall", 
                      "num_trials", "message"]
        
        for key in metric_keys:
            if key in response_data:
                metrics[key] = response_data[key]
        
        if metrics:
            logger.info(f"Response metrics: {metrics}")
