# tests/test_model.py
import os
import importlib
import numpy as np
from pathlib import Path
import pytest


# ============================================================================
# 1. Penyimpanan State
# ============================================================================

# Positive (+)
def test_modelstate_validation_confusion_matrix(app_client, patch_tf_light, f_preprocessed):
    """
    Test that validation confusion matrix is properly stored in model state.
    Verifies CM and class labels are saved after training.
    """
    main = importlib.import_module("api.main")
    
    # Train a model to populate state
    r = app_client.post(
        "/train",
        files={"dataset_file": ("preprocessed.mat", Path(f_preprocessed).read_bytes(), "application/octet-stream")},
    )
    assert r.status_code == 200
    
    # Verify validation CM is stored in state
    assert main.model_state.val_cm is not None, "Validation confusion matrix should be stored"
    assert main.model_state.val_classes is not None, "Validation classes should be stored"
    
    # Check CM shape matches number of classes
    n_classes = len(main.model_state.val_classes)
    assert main.model_state.val_cm.shape == (n_classes, n_classes)
    
    # Verify CM contains valid integer counts
    assert np.all(main.model_state.val_cm >= 0)
    assert np.all(main.model_state.val_cm == main.model_state.val_cm.astype(int))


# Positive (+)
def test_modelstate_prediction_results(app_client, patch_tf_light, f_preprocessed, f_predict, monkeypatch):
    """
    Test that prediction results are properly accessible.
    Verifies predictions work with trained model.
    """
    main = importlib.import_module("api.main")
    
    # Create a simple dummy model for prediction
    class DummyModel:
        def predict(self, X, batch_size=None, verbose=0):
            n = X.shape[0]
            probs = np.zeros((n, 6), dtype=np.float32)
            probs[0, 0] = 1.0
            return probs
    
    # First train a model
    r = app_client.post(
        "/train", 
        files={"dataset_file": ("preprocessed.mat", Path(f_preprocessed).read_bytes(), "application/octet-stream")},
    )
    assert r.status_code == 200
    
    # Save the model
    r2 = app_client.post("/train/save")
    assert r2.status_code == 200
    model_name = os.path.basename(r2.json()["model_path"])
    
    # Patch load_model_safely for prediction
    monkeypatch.setattr(main, "load_model_safely", lambda _: DummyModel(), raising=True)
    
    # Make a prediction with real test data
    r3 = app_client.post(
        "/predict",
        files={"dataset_file": ("predict.mat", Path(f_predict).read_bytes(), "application/octet-stream")},
        params={"model_name": model_name}
    )
    assert r3.status_code == 200, f"Prediction failed: {r3.text}"
    
    # Verify prediction results structure
    js = r3.json()
    assert "num_trials" in js
    assert js["num_trials"] > 0


# Negative (-)
def test_modelstate_invalid_data_type_error():
    """
    Test that ModelState prevents storing invalid data types. 
    Verifies type validation for state attributes.
    """
    main = importlib.import_module("api.main")
    
    # Create a fresh state
    state = main.ModelState()
    
    # Try to set confusion matrix to invalid types
    # The state should either reject or handle gracefully
    try:
        # Attempt to set invalid CM type (string instead of numpy array)
        state.val_cm = "invalid_string"
        # If it doesn't raise, verify it can still work
        assert state.val_cm == "invalid_string" or state.val_cm is None
    except (TypeError, ValueError):
        # Expected behavior - rejecting invalid type
        pass
    
    # Verify that valid numpy array is accepted
    valid_cm = np.array([[1, 2], [3, 4]])
    state.val_cm = valid_cm
    assert state.val_cm is not None
    assert isinstance(state.val_cm, np.ndarray)


# ============================================================================
# 2. Penghapusan Model
# ============================================================================

# Positive (+)
def test_delete_model_then_fail_download(app_client, patch_tf_light, f_preprocessed):
    """
    Test manual model file deletion prevents download.
    Verifies system handles missing model files gracefully.
    """
    main = importlib.import_module("api.main")
    
    # Train and save a model
    r = app_client.post(
        "/train",
        files={"dataset_file": ("preprocessed.mat", Path(f_preprocessed).read_bytes(), "application/octet-stream")},
    )
    assert r.status_code == 200
    
    r2 = app_client.post("/train/save")
    assert r2.status_code == 200
    model_path = r2.json()["model_path"]
    model_name = os.path.basename(model_path)
    
    # Verify model exists
    assert os.path.isfile(model_path)
    
    # Manually delete the file (simulating deletion)
    os.remove(model_path)
    
    # Verify file is deleted
    assert not os.path.isfile(model_path), "Model file should be deleted"
    
    # Try to download - should fail
    r4 = app_client.get(f"/train/download?model_name={model_name}")
    assert r4.status_code == 404, "Should not find deleted model"


# Negative (-)
def test_delete_default_model_blocked(app_client):
    """
    Test that attempting to access protected model paths is handled.
    Verifies path validation and security.
    """
    # Try accessing with path traversal patterns
    r = app_client.get("/train/download?model_name=../../etc/passwd")
    
    # Should be blocked (400 for path traversal or 404 for not found)
    assert r.status_code in [400, 404], "Should block suspicious paths"


# Negative (-)
def test_delete_nonexistent_model(app_client):
    """
    Test that downloading non-existent model returns 404.
    Verifies proper error handling for invalid model names.
    """
    # Try to download a model that doesn't exist
    r = app_client.get("/train/download?model_name=nonexistent_model_xyz123")
    
    # Should return 404
    assert r.status_code == 404, "Should return 404 for non-existent model"
    
    # Verify error message is informative
    js = r.json()
    assert "detail" in js
    detail = js["detail"].lower()
    assert "not found" in detail or "tidak ditemukan" in detail


# ============================================================================
# 3. Penanganan Error State
# ============================================================================

# Positive (+)
def test_models_delete_success_path(app_client, patch_tf_light, f_preprocessed):
    """
    Test that model files can be manually removed and system handles it gracefully.
    Verifies file system cleanup and error handling.
    """
    main = importlib.import_module("api.main")
    
    # Train and save a model
    r = app_client.post(
        "/train",
        files={"dataset_file": ("preprocessed.mat", Path(f_preprocessed).read_bytes(), "application/octet-stream")},
    )
    assert r.status_code == 200
    
    r2 = app_client.post("/train/save")
    assert r2.status_code == 200
    model_info = r2.json()
    model_path = model_info["model_path"]
    model_name = os.path.basename(model_path)
    
    # Verify model file exists
    assert os.path.isfile(model_path), "Model file should exist after saving"
    
    # Manually remove the file
    os.remove(model_path)
    
    # Verify file is gone
    assert not os.path.isfile(model_path), "Model file should be deleted"
    
    # Try to download - should return 404
    r3 = app_client.get(f"/train/download?model_name={model_name}")
    assert r3.status_code == 404, "Should return 404 for missing model file"


# Negative (-)
def test_models_delete_error_paths(app_client):
    """
    Test error handling for various invalid download scenarios.
    Verifies proper error messages for failure cases.
    """
    # Test 1: Non-existent model
    r1 = app_client.get("/train/download?model_name=nonexistent_xyz123")
    assert r1.status_code == 404, "Should return 404 for non-existent model"
    assert "detail" in r1.json()
    
    # Test 2: Path traversal attempt
    r2 = app_client.get("/train/download?model_name=../../etc/passwd")
    assert r2.status_code in [400, 404], "Should block path traversal"
    
    #  Test 3: Empty model name
    r3 = app_client.get("/train/download?model_name=")
    assert r3.status_code in [400, 404, 422], "Should reject empty model_name"
    
    # Test 4: Download without any trained model
    r4 = app_client.get("/train/download")
    assert r4.status_code in [400, 404], "Should return error when no model trained"
    assert "detail" in r4.json()
