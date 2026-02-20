# tests/test_predict.py
import os
import importlib
import numpy as np
from pathlib import Path

def _touch_dummy_model_file() -> str:
    """
    Create a dummy .h5 file in the models directory.
    Returns the filename (not full path).
    """
    main = importlib.import_module("api.main")
    
    # Create file in MODELS_DIR instead of temp directory
    test_model_name = f"test_model_{int(np.random.rand() * 10000)}.h5"
    model_path = os.path.join(main.MODELS_DIR, test_model_name)
    
    with open(model_path, 'wb') as f:
        f.write(b"DUMMY")
    
    return test_model_name  # Return just the filename, not the full path


# ============================================================================
# 1. Validasi Data Input
# ============================================================================

# Positive (+)
def test_predict_valid_data_structure_accepted(app_client, make_mat_bytes, monkeypatch):
    """
    Test that prediction accepts data with complete variable structure.
    Ensures valid data with proper shape and 'data' variable is accepted.
    """
    main = importlib.import_module("api.main")

    class DummyModel:
        def predict(self, X, batch_size=None, verbose=0):
            n = X.shape[0]
            probs = np.zeros((n, 6), dtype=np.float32)
            for i in range(n):
                probs[i, i % 6] = 1.0
            return probs

    monkeypatch.setattr(main, "load_model_safely", lambda _: DummyModel(), raising=True)
    model_name = _touch_dummy_model_file()

    # Valid data: (10, 64) -> divisible by 5, has 'data' variable
    valid_data = np.random.randn(10, 64).astype("float32")
    mat = make_mat_bytes({"data": valid_data})
    files = {"dataset_file": ("predict.mat", mat, "application/octet-stream")}
    
    r = app_client.post("/predict", files=files, params={"model_name": model_name})
    
    # Should succeed
    assert r.status_code == 200, r.text
    js = r.json()
    
    # Check response structure (actual API response keys)
    assert "num_trials" in js
    assert "isSingleSample" in js
    assert "message" in js
    
    # Verify basic structure
    assert isinstance(js["num_trials"], int)
    assert js["num_trials"] > 0


# Negative (-)
def test_predict_missing_data_variable_returns_400(app_client, make_mat_bytes, monkeypatch):
    """sio.loadmat row: ensure 'data' key is required."""
    main = importlib.import_module("api.main")

    class DummyModel:
        def predict(self, X, batch_size=None, verbose=0):  # pragma: no cover
            return np.zeros((X.shape[0], 6), dtype=np.float32)

    monkeypatch.setattr(main, "load_model_safely", lambda _: DummyModel(), raising=True)
    model_name = _touch_dummy_model_file()

    bad = make_mat_bytes({"not_data": np.zeros((5, 64), dtype="float32")})
    files = {"dataset_file": ("predict.mat", bad, "application/octet-stream")}
    r = app_client.post("/predict", files=files, params={"model_name": model_name})
    assert r.status_code == 400
    assert "data" in r.json()["detail"].lower()


# Negative (-)
def test_predict_wrong_feature_dim_returns_400(app_client, make_mat_bytes, monkeypatch):
    """2D input with second dim != 64 -> 400 mentioning expected 64."""
    main = importlib.import_module("api.main")

    class DummyModel:
        def predict(self, X, batch_size=None, verbose=0):
            n = X.shape[0]
            probs = np.zeros((n, 6), dtype=np.float32)
            for i in range(n):
                probs[i, i % 6] = 1.0
            return probs

    monkeypatch.setattr(main, "load_model_safely", lambda _: DummyModel(), raising=True)
    model_name = _touch_dummy_model_file()

    bad = make_mat_bytes({"data": (np.random.randn(5, 63)).astype("float32")})
    files = {"dataset_file": ("invalid.mat", bad, "application/octet-stream")}
    r = app_client.post("/predict", files=files, params={"model_name": model_name})
    assert r.status_code == 400
    assert "64" in r.json()["detail"]


# Negative (-)
def test_predict_bad_shape_returns_400(app_client, make_mat_bytes, monkeypatch):
    """(n,64) where n is not divisible by 5 -> 400 with helpful message."""
    main = importlib.import_module("api.main")

    class DummyModel:
        def predict(self, X, batch_size=None, verbose=0):
            n = X.shape[0]
            probs = np.zeros((n, 6), dtype=np.float32)
            for i in range(n):
                probs[i, i % 6] = 1.0
            return probs

    monkeypatch.setattr(main, "load_model_safely", lambda _: DummyModel(), raising=True)
    model_name = _touch_dummy_model_file()

    bad = make_mat_bytes({"data": (main.np.random.randn(7, 64)).astype("float32")})
    files = {"dataset_file": ("invalid.mat", bad, "application/octet-stream")}
    r = app_client.post("/predict", files=files, params={"model_name": model_name})
    assert r.status_code == 400
    assert "divisible by 5" in r.json()["detail"].lower()


# ============================================================================
# 2. Keamanan Model Path
# ============================================================================

# Positive (+)
def test_predict_valid_model_path_access(app_client, make_mat_bytes, monkeypatch):
    """
    Test that prediction allows access to valid model paths.
    Ensures legitimate model names within the models directory are accepted.
    """
    main = importlib.import_module("api.main")

    class DummyModel:
        def predict(self, X, batch_size=None, verbose=0):
            n = X.shape[0]
            probs = np.zeros((n, 6), dtype=np.float32)
            probs[0, 0] = 1.0
            return probs

    monkeypatch.setattr(main, "load_model_safely", lambda _: DummyModel(), raising=True)
    
    # Create a dummy model with a normal, valid name
    model_name = _touch_dummy_model_file()

    # Valid data
    X = np.random.randn(5, 64).astype("float32")
    mat = make_mat_bytes({"data": X})
    files = {"dataset_file": ("predict.mat", mat, "application/octet-stream")}
    
    # Should accept valid model name
    r = app_client.post("/predict", files=files, params={"model_name": model_name})
    assert r.status_code == 200, r.text
    
    # Also test without .h5 extension (server should add it)
    model_name_no_ext = model_name.replace(".h5", "")
    r2 = app_client.post("/predict", files=files, params={"model_name": model_name_no_ext})
    assert r2.status_code == 200, r2.text


# Negative (-)
def test_predict_with_invalid_model_name(app_client, f_dataset_awal):
    """Test prediction with invalid model name."""
    files = {
        "dataset_file": ("predict.mat", Path(f_dataset_awal).read_bytes(), "application/octet-stream"),
    }
    response = app_client.post("/predict?model_name=../../../etc/passwd", files=files)
    assert response.status_code == 400
    assert "path traversal" in response.json()["detail"].lower()


# Negative (-)
def test_predict_with_nonexistent_model(app_client, f_dataset_awal):
    """Test prediction with non-existent model."""
    files = {
        "dataset_file": ("predict.mat", Path(f_dataset_awal).read_bytes(), "application/octet-stream"),
    }
    response = app_client.post("/predict?model_name=nonexistent_model", files=files)
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


# ============================================================================
# 3. Logika Pemrosesan
# ============================================================================

# Positive (+)
def test_predict_single_sample_sets_flag_true(app_client, make_mat_bytes, monkeypatch):
    """data.ndim row: exactly (5,64) -> single-sample mode."""
    main = importlib.import_module("api.main")

    class DummyModel:
        def predict(self, X, batch_size=None, verbose=0):
            probs = np.zeros((X.shape[0], 6), dtype=np.float32)
            probs[0, 0] = 1.0
            return probs

    monkeypatch.setattr(main, "load_model_safely", lambda _: DummyModel(), raising=True)
    model_name = _touch_dummy_model_file()

    X = (np.random.randn(5, 64)).astype("float32")
    mat = make_mat_bytes({"data": X})
    files = {"dataset_file": ("predict.mat", mat, "application/octet-stream")}
    r = app_client.post("/predict", files=files, params={"model_name": model_name})
    assert r.status_code == 200
    js = r.json()
    assert js["num_trials"] == 1
    assert js["isSingleSample"] is True


# Positive (+)
def test_predict_2d_multiple_trials_reshape_ok(app_client, make_mat_bytes, monkeypatch):
    """data.ndim row: (n,64) with n%5==0 reshapes to (n/5,5,64)."""
    main = importlib.import_module("api.main")

    class DummyModel:
        def predict(self, X, batch_size=None, verbose=0):
            n = X.shape[0]
            probs = np.zeros((n, 6), dtype=np.float32)
            for i in range(n):
                probs[i, i % 6] = 1.0
            return probs

    monkeypatch.setattr(main, "load_model_safely", lambda _: DummyModel(), raising=True)
    model_name = _touch_dummy_model_file()

    X = (np.random.randn(10, 64)).astype("float32")  # -> 2 trials
    mat = make_mat_bytes({"data": X})
    files = {"dataset_file": ("predict.mat", mat, "application/octet-stream")}
    r = app_client.post("/predict", files=files, params={"model_name": model_name})
    assert r.status_code == 200, r.text
    js = r.json()
    assert js["num_trials"] == 2
    assert js["isSingleSample"] is False
    assert len(js["time_windows"]) == 2
    assert js["time_windows"][0]["start"] == 0 and js["time_windows"][0]["end"] == 5


# ============================================================================
# 4. Utilitas File
# ============================================================================

# Positive (+)
def test_extract_file_id_with_various_formats():
    """Test extract_file_id with different filename formats."""
    main = importlib.import_module("api.main")
    
    # Test standard underscore format
    assert main.extract_file_id("data_12345.mat") == "12345"
    
    # Test participant format
    assert main.extract_file_id("dataP123.mat") == "P123"
    
    # Test with no recognizable pattern
    file_id = main.extract_file_id("nopattern.mat")
    assert isinstance(file_id, str)
    assert len(file_id) > 0  # Should generate a timestamp
    
    # Test with None
    file_id = main.extract_file_id(None)
    assert isinstance(file_id, str)
    assert len(file_id) > 0


# Negative (-)
def test_extract_file_id_returns_none_on_invalid():
    """
    Test that extract_file_id returns a fallback value (timestamp or None) 
    when given invalid or unrecognized filename formats.
    """
    main = importlib.import_module("api.main")
    
    # Test with empty string
    result = main.extract_file_id("")
    # Should either return None or a valid timestamp string
    assert result is None or (isinstance(result, str) and len(result) > 0)
    
    # Test with special characters only
    result = main.extract_file_id("!@#$%^&*()")
    # Should handle gracefully
    assert result is None or (isinstance(result, str) and len(result) > 0)
    
    # Test with very long invalid string
    result = main.extract_file_id("x" * 1000)
    # Should handle gracefully
    assert result is None or (isinstance(result, str) and len(result) > 0)
    
    # Test with path-like invalid string (but not actual path traversal)
    result = main.extract_file_id("random/path/name.txt")
    # Should extract something or return fallback
    assert result is None or (isinstance(result, str) and len(result) > 0)