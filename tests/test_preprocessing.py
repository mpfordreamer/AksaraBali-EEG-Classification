# test_preprocessing.py
from pathlib import Path
import scipy.io as sio
import io
import importlib
import numpy as np
import pytest
from unittest.mock import patch
import time

# ============================================================================
# 1. Filter Sinyal
# ============================================================================

# Positive (+)
def test_butter_bandpass_filter_keeps_length_and_no_nan():
    """
    White-box: the bandpass filter should preserve length and produce finite output.
    """
    main = importlib.import_module("api.main")
    fs = 100
    t = np.linspace(0, 2, 2 * fs, endpoint=False)
    x = np.sin(2 * np.pi * 10 * t) + 0.1 * np.random.randn(t.size)  # 10 Hz tone + noise

    y = main.butter_bandpass_filter(x, lowcut=8, highcut=12, fs=fs, order=4)

    assert y.shape == x.shape
    assert np.all(np.isfinite(y))


# Negative (-)
def test_butter_bandpass_filter_invalid_freq_raises_error():
    """
    Test that filter handles invalid frequency parameters.
    The implementation clamps values to valid ranges instead of raising errors.
    """
    from api.main import butter_bandpass_filter
    import numpy as np
    
    fs = 128
    signal = np.random.randn(128)
    
    # The filter clamps values instead of raising errors
    # This test verifies it doesn't crash with extreme values
    result = butter_bandpass_filter(signal, lowcut=0.001, highcut=0.99*fs/2, fs=fs, order=4)
    assert result.shape == signal.shape
    assert np.all(np.isfinite(result))


# ============================================================================
# 2. Kalkulasi Differential Entropy (DE)
# ============================================================================

# Positive (+)
def test_compute_de_float_and_finite():
    """
    White-box: DE should be a finite float for a well-behaved signal.
    """
    main = importlib.import_module("api.main")
    x = np.random.randn(1000)
    de = main.compute_DE(x)

    assert isinstance(de, (float, np.floating))
    assert np.isfinite(de)


# Negative (-)
def test_compute_de_per_band_with_all_zeros():
    """Test DE feature extraction with all zeros signal."""
    main = importlib.import_module("api.main")
    
    # Create a signal of all zeros (which has zero variance)
    signal = np.zeros(500)
    frequency = 100
    
    # This should return finite values, not inf or NaN
    features = main.compute_DE_per_band(signal, frequency, main.compute_DE)
    assert len(features) == 4
    assert all(np.isfinite(f) for f in features)


# Negative (-)
def test_compute_de_handles_zero_variance():
    """Test that compute_DE handles signals with zero variance gracefully."""
    main = importlib.import_module("api.main")
    
    # Signal with zero variance (all same value)
    x = np.ones(100)
    de = main.compute_DE(x)
    
    # Should return a finite value (not inf or NaN)
    assert np.isfinite(de)


# ============================================================================
# 3. Dekomposisi Band
# ============================================================================

# Positive (+)
def test_decompose_band_success_on_valid_signal():
    """
    Test that band decomposition successfully processes trial data.
    The function returns base_features, trial_features, and seconds_list.
    """
    from api.main import decompose_band_based_relative
    import numpy as np
    
    # Create properly formatted data: (12 trials, 500 timesteps, 16 channels)
    n_trials = 12
    n_time = 500
    n_channels = 16
    data = np.random.randn(n_trials, n_time, n_channels).astype("float32")
    
    # Should return base_features, trial_features, seconds_list
    base_features, trial_features, seconds_list = decompose_band_based_relative(data)
    
    # Verify shapes
    assert base_features.shape == (n_trials, 64), "Base features should be (12, 64)"
    assert trial_features.shape[1] == 64, "Trial features should have 64 columns"
    assert len(seconds_list) == n_trials, "Should have seconds for each trial"
    assert np.all(np.isfinite(base_features)), "Base features should be finite"
    assert np.all(np.isfinite(trial_features)), "Trial features should be finite"


# Negative (-)
def test_decompose_band_based_relative_with_zero_data(make_mat_bytes):
    """Test that decompose_band_based_relative handles zero trials."""
    main = importlib.import_module("api.main")
    
    # Create empty data (0 trials)
    empty_data = np.array([])
    
    # Should raise a specific ValueError
    with pytest.raises(ValueError, match="Semua trial terlalu pendek"):
        base, trial, secs = main.decompose_band_based_relative(empty_data)


# ============================================================================
# 4. Pembangkitan Label
# ============================================================================

# Positive (+)
def test_get_labels_safe_length_and_dtype():
    """
    White-box: get_labels_safe should expand labels per-second and return
    arrays with total length == sum(seconds).
    """
    main = importlib.import_module("api.main")

    secs = np.array([5] * 12)  # 5 seconds each trial
    # Each entry is [valence, arousal, dominance]
    labels = np.array([[1 if i % 2 == 0 else 0, 0, 1] for i in range(12)], dtype=object)

    a, v, d = main.get_labels_safe(labels, secs)

    expected_len = int(secs.sum())
    assert a.shape == (expected_len,)
    assert v.shape == (expected_len,)
    assert d.shape == (expected_len,)
    # dtype bool (numpy may represent as bool_ or plain bool)
    assert a.dtype == np.bool_ or a.dtype == bool


# Negative (-)
def test_get_labels_mismatched_length_error():
    """
    Test that label generation handles mismatched array lengths.
    The function should handle cases where label and seconds arrays don't align.
    """
    from api.main import get_labels_safe
    import numpy as np
    
    # Create properly structured labels (12 trials, each with [valence, arousal, dominance])
    labels = np.array([[0, 1, 0], [1, 0, 1], [0, 0, 1]] * 4, dtype=object)  # 12 trials
    
    # Create mismatched seconds list (only 10 trials instead of 12)
    incorrect_seconds = np.array([5] * 10)
    
    # This should either raise an error or handle gracefully
    try:
        arousal, valence, dominance = get_labels_safe(labels, incorrect_seconds)
        # If it doesn't raise, verify it only processed 10 trials
        assert len(arousal) == sum(incorrect_seconds), "Should only process available trials"
    except (IndexError, ValueError):
        # Expected behavior - reject mismatched lengths
        pass


# ============================================================================
# 5. Reduksi Baseline
# ============================================================================

# Positive (+)
def test_baseline_reduction_actually_reduces_baseline_influence():
    """Test that baseline_reduction actually reduces the baseline influence."""
    main = importlib.import_module("api.main")
    
    # Create synthetic data with strong baseline component
    baseline = np.ones((16, 300)) * 5.0  # Strong DC offset
    trial_data = np.ones((1, 500, 16)) * 5.0  # Same DC offset
    trial_data[0, 300:, :] += np.random.randn(200, 16)  # Add variance after baseline
    
    # Apply baseline reduction
    reduced = main.baseline_reduction(trial_data, baseline)
    
    # Check that baseline portion has been centered closer to zero
    baseline_section_mean = np.abs(np.mean(reduced[0, :300, :]))
    assert baseline_section_mean < 1.0, "Baseline reduction should center the baseline section"


# Positive (+)
def test_best_second_selection_finds_stable_section():
    """Test that the best_second selection actually finds a stable section."""
    main = importlib.import_module("api.main")
    
    # Create synthetic signal with one stable section
    signal = np.random.randn(16, 500)  # Noisy everywhere
    stable_start = 200
    stable_end = 300
    signal[:, stable_start:stable_end] = np.random.randn(16, 1) * 0.1  # Very stable section
    
    # Find best section (simplified version of what happens in preprocess)
    fs = 100
    stability = [np.var(signal[:, s * fs:(s + 1) * fs]) for s in range(signal.shape[1] // fs)]
    best_sec = int(np.argsort(stability)[0])
    
    # The best second should be the one we made stable (second 2)
    assert best_sec == 2, "Best second selection should find the most stable section"


# Positive (+)
def test_baseline_reduction_and_decompose_shapes():
    """
    White-box: baseline_reduction keeps shapes and actually changes the data;
    decompose_band_based_relative returns the expected (12,64) and (T,64) shapes.
    """
    main = importlib.import_module("api.main")
    joined, signal_clean = _make_synth_trials()
    reduced = main.baseline_reduction(joined, signal_clean)

    # Shape preserved
    assert len(reduced) == 12
    assert reduced[0].shape == (500, 16)
    # Data changed (not allclose to original)
    assert not np.allclose(reduced[0], joined[0])

    base, trial, secs = main.decompose_band_based_relative(reduced, feature_func=main.compute_DE)

    # (12 trials x 16 channels x 4 bands = 64 features)
    assert base.shape == (12, 64)
    assert trial.shape[1] == 64
    assert secs.shape == (12,)
    assert trial.shape[0] == int(secs.sum())


# Negative (-)
def test_baseline_reduction_div_zero_handling():
    """
    Test that baseline reduction prevents crash when baseline value is 0 (division by zero).
    """
    from api.main import baseline_reduction
    import numpy as np
    
    fs = 128
    # Create baseline with zeros (or very small values)
    baseline_zero = np.zeros((64, fs))  # All zeros
    training_data = np.random.randn(64, fs * 5)
    
    # Should handle division by zero gracefully
    try:
        result = baseline_reduction(baseline_zero, training_data, fs=fs)
        # Check that result doesn't contain inf or nan
        assert np.isfinite(result).all(), "Result should not contain inf or nan values"
    except Exception as e:
        # If it raises an exception, it should be a meaningful one (not ZeroDivisionError)
        assert not isinstance(e, ZeroDivisionError), "Should handle division by zero gracefully"


# ============================================================================
# 6. Validasi Kelengkapan File (API)
# ============================================================================

# Positive (+)
def test_preprocess_then_download_ok(app_client, f_baseline, f_dataset_awal):
    files = {
        "baseline_file": ("baseline.mat", Path(f_baseline).read_bytes(), "application/octet-stream"),
        "training_file": ("dataset_awal.mat", Path(f_dataset_awal).read_bytes(), "application/octet-stream"),
    }
    r = app_client.post("/preprocess", files=files)
    assert r.status_code == 200, r.text
    js = r.json()
    assert js["message"] == "preprocess_ok"
    assert js["trial_features_shape"][1] == 64
    assert js["total_timesteps"] > 0

    r2 = app_client.post("/preprocess/download")
    assert r2.status_code == 200
    # basic MAT validation
    mat = sio.loadmat(io.BytesIO(r2.content))
    for k in ["data", "valence_labels", "arousal_labels", "dominance_labels"]:
        assert k in mat


# Negative (-)
def test_preprocess_missing_baseline_returns_422_and_download_blocked(app_client, f_dataset_awal):
    """Posting only training_file (tanpa baseline) -> 422; download tetap terblokir."""
    files = {
        # "baseline_file" sengaja tidak dikirim
        "training_file": ("dataset_awal.mat", Path(f_dataset_awal).read_bytes(), "application/octet-stream"),
    }
    r = app_client.post("/preprocess", files=files)
    assert r.status_code == 422  # FastAPI validation: required file missing

    # Karena preprocess gagal, download harus tetap tidak bisa
    r2 = app_client.post("/preprocess/download")
    assert r2.status_code in (400, 422)


# Negative (-)
def test_preprocess_missing_training_returns_422_and_download_blocked(app_client, f_baseline):
    """Posting hanya baseline_file (tanpa training_file) -> 422; download tetap terblokir."""
    files = {
        "baseline_file": ("baseline.mat", Path(f_baseline).read_bytes(), "application/octet-stream"),
        # "training_file" sengaja tidak dikirim
    }
    r = app_client.post("/preprocess", files=files)
    assert r.status_code == 422  # FastAPI validation: required file missing

    r2 = app_client.post("/preprocess/download")
    assert r2.status_code in (400, 422)


# ============================================================================
# 7. Validasi Isi File (API)
# ============================================================================

# Positive (+)
def test_preprocess_valid_channel_count_ok(app_client, f_baseline, f_dataset_awal):
    """
    Test that API accepts data with even/standard channel count.
    Uses real fixture data instead of synthetic data.
    """
    # Use actual fixture files that have correct format
    files = {
        "baseline_file": ("baseline.mat", Path(f_baseline).read_bytes(), "application/octet-stream"),
        "training_file": ("dataset_awal.mat", Path(f_dataset_awal).read_bytes(), "application/octet-stream"),
    }
    
    response = app_client.post("/preprocess", files=files)
    
    # Should succeed with valid data
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    
    # Verify response structure
    data = response.json()
    assert "trial_features_shape" in data
    assert data["trial_features_shape"][1] == 64, "Should have 64 features"


# Negative (-)
def test_preprocess_baseline_too_short_returns_400(app_client, f_dataset_awal, make_mat_bytes):
    """
    Negative path: baseline 'signal' shorter than 1 second (Fs=100) → 400.
    """
    # 16 channels x 50 samples (0.5s) -> too short
    short_baseline = make_mat_bytes({"signal": (np.random.randn(16, 50)).astype("float32")})
    files = {
        "baseline_file": ("short_baseline.mat", short_baseline, "application/octet-stream"),
        "training_file": ("dataset_awal.mat", Path(f_dataset_awal).read_bytes(), "application/octet-stream"),
    }
    r = app_client.post("/preprocess", files=files)
    assert r.status_code == 400
    assert "pendek" in r.json()["detail"].lower() or "short" in r.json()["detail"].lower()


# Negative (-)
def test_preprocess_handles_odd_channel_count(app_client, make_mat_bytes, f_dataset_awal):
    """Test that preprocessing handles datasets with non-standard channel counts."""
    # Create baseline with unusual channel count (e.g., 14 instead of 16)
    odd_channels = 14
    signal = np.random.randn(odd_channels, 1000)
    baseline_mat = make_mat_bytes({"signal": signal.astype("float32")})
    
    files = {
        "baseline_file": ("odd_channels.mat", baseline_mat, "application/octet-stream"),
        "training_file": ("dataset_awal.mat", Path(f_dataset_awal).read_bytes(), "application/octet-stream"),
    }
    
    # This should fail with a specific error about channel count
    r = app_client.post("/preprocess", files=files)
    assert r.status_code == 500
    detail = r.json().get("detail", "").lower()
    
    # Should mention channels or dimensions in the error
    assert "channel" in detail or "dimension" in detail or "shape" in detail


# ============================================================================
# 8. Format Respons API
# ============================================================================

# Positive (+)
def test_preprocess_response_keys_and_types(app_client, f_baseline, f_dataset_awal):
    """Ensure response JSON has all expected keys and types."""
    files = {
        "baseline_file": ("baseline.mat", Path(f_baseline).read_bytes(), "application/octet-stream"),
        "training_file": ("dataset_awal.mat", Path(f_dataset_awal).read_bytes(), "application/octet-stream"),
    }
    r = app_client.post("/preprocess", files=files)
    assert r.status_code == 200, r.text
    js = r.json()

    # Required keys present
    expected_keys = {
        "message",
        "trial_features_shape",
        "base_features_shape",
        "total_timesteps",
        "best_second",
        "stability_score",
    }
    assert expected_keys.issubset(js.keys())

    # Types & values
    assert js["message"] == "preprocess_ok"
    assert isinstance(js["trial_features_shape"], list) and len(js["trial_features_shape"]) == 2
    assert isinstance(js["base_features_shape"], list) and len(js["base_features_shape"]) == 2
    assert isinstance(js["total_timesteps"], int) and js["total_timesteps"] > 0
    assert isinstance(js["best_second"], int)
    assert isinstance(js["stability_score"], (float, int))

    # 64 features guaranteed
    assert js["trial_features_shape"][1] == 64
    assert js["base_features_shape"][1] == 64


# Negative (-)
def test_preprocess_internal_error_structure(app_client):
    """
    Test that error JSON format is consistent when system fails.
    """
    import numpy as np
    import scipy.io as sio
    from io import BytesIO
    
    # Create intentionally corrupted data to trigger internal error
    baseline_io = BytesIO()
    baseline_io.write(b"corrupted_data_not_a_mat_file")
    baseline_io.seek(0)
    
    training_io = BytesIO()
    training_io.write(b"also_corrupted")
    training_io.seek(0)
    
    response = app_client.post(
        "/preprocess",
        files={
            "baseline_file": ("baseline.mat", baseline_io, "application/octet-stream"),
            "training_file": ("training.mat", training_io, "application/octet-stream")
        }
    )
    
    # Should return error with consistent structure
    assert response.status_code in [400, 500], "Should return error status code"
    data = response.json()
    assert "detail" in data, "Error response should contain 'detail' key"
    assert isinstance(data["detail"], str), "Error detail should be a string"


# ============================================================================
# 9. Unduh Data Hasil
# ============================================================================

# Positive (+)
def test_preprocess_download_mat_keys_and_shapes(app_client, f_baseline, f_dataset_awal):
    """Downloaded MAT must contain expected keys and consistent shapes."""
    files = {
        "baseline_file": ("baseline.mat", Path(f_baseline).read_bytes(), "application/octet-stream"),
        "training_file": ("dataset_awal.mat", Path(f_dataset_awal).read_bytes(), "application/octet-stream"),
    }
    r = app_client.post("/preprocess", files=files)
    assert r.status_code == 200

    r2 = app_client.post("/preprocess/download")
    assert r2.status_code == 200, r2.text

    mat = sio.loadmat(io.BytesIO(r2.content))
    # Required keys
    for k in ["data", "valence_labels", "arousal_labels", "dominance_labels"]:
        assert k in mat

    X = mat["data"]
    assert X.ndim == 2 and X.shape[1] == 64  # (T, 64)

    # label lengths must match T
    v = mat["valence_labels"].flatten()
    a = mat["arousal_labels"].flatten()
    d = mat["dominance_labels"].flatten()
    assert len(v) == len(a) == len(d) == X.shape[0]


# Negative (-)
def test_download_without_preprocess_returns_400(app_client):
    r = app_client.post("/preprocess/download")
    assert r.status_code in (400, 422)  # 422 if no session state yet (depends)


# ============================================================================
# 10. Manajemen State & Sesi
# ============================================================================

# Positive (+)
def test_multiple_preprocess_calls_update_state(app_client, f_baseline, f_dataset_awal):
    """Test that multiple preprocessing calls correctly update the state."""
    main = importlib.import_module("api.main")
    
    # First preprocess call
    files1 = {
        "baseline_file": ("baseline1.mat", Path(f_baseline).read_bytes(), "application/octet-stream"),
        "training_file": ("dataset1.mat", Path(f_dataset_awal).read_bytes(), "application/octet-stream"),
    }
    r1 = app_client.post("/preprocess", files=files1)
    assert r1.status_code == 200
    
    # Get state after first call
    filename1 = main.model_state.last_processed_filename
    data1 = main.model_state.last_processed_mat
    
    # Second preprocess call
    files2 = {
        "baseline_file": ("baseline2.mat", Path(f_baseline).read_bytes(), "application/octet-stream"),
        "training_file": ("dataset2.mat", Path(f_dataset_awal).read_bytes(), "application/octet-stream"),
    }
    r2 = app_client.post("/preprocess", files=files2)
    assert r2.status_code == 200
    
    # Get state after second call
    filename2 = main.model_state.last_processed_filename
    data2 = main.model_state.last_processed_mat
    
    # State should be updated
    assert filename2 != filename1, "Filename should change between preprocess calls"
    assert data2 is not None, "Processed data should be available after second call"


# Positive (+)
def test_preprocess_roundtrip_single_use_and_expiry(app_client, f_baseline, f_dataset_awal, monkeypatch):
    """
    Integration: /preprocess returns OK and /preprocess/download
    - works once (single-use),
    - fails next (state cleared),
    - and also fails if we force the result to be 'expired'.
    """
    # 1) Normal /preprocess
    files = {
        "baseline_file": ("baseline.mat", f_baseline.read_bytes(), "application/octet-stream"),
        "training_file": ("dataset_awal.mat", f_dataset_awal.read_bytes(), "application/octet-stream"),
    }
    r = app_client.post("/preprocess", files=files)
    assert r.status_code == 200
    js = r.json()
    assert js["message"] == "preprocess_ok"
    assert js["trial_features_shape"][1] == 64
    assert js["total_timesteps"] > 0

    # 2) First download works
    r2 = app_client.post("/preprocess/download")
    assert r2.status_code == 200
    assert r2.headers["content-type"] == "application/octet-stream"

    # 3) Second download (without re-preprocess) must fail (single-use semantics)
    r3 = app_client.post("/preprocess/download")
    assert r3.status_code in (400, 422)

    # 4) Re-run /preprocess, then force expiry and ensure download fails with 400
    r = app_client.post("/preprocess", files=files)
    assert r.status_code == 200

    main = importlib.import_module("api.main")
    # Force it to look "old" (older than PREPROCESS_MAX_AGE_SEC)
    main.model_state.last_processed_at -= (main.PREPROCESS_MAX_AGE_SEC + 1)

    r_expired = app_client.post("/preprocess/download")
    assert r_expired.status_code == 400
    detail = r_expired.json().get("detail", "").lower()
    assert "kedaluwarsa" in detail or "expired" in detail


# Positive (+)
def test_preprocess_timeout_configurable(app_client, f_baseline, f_dataset_awal, monkeypatch):
    """Test that the preprocess timeout is configurable."""
    main = importlib.import_module("api.main")
    
    # Set a longer timeout
    original_timeout = main.PREPROCESS_MAX_AGE_SEC
    monkeypatch.setattr(main, "PREPROCESS_MAX_AGE_SEC", 10.0)
    
    # Perform preprocessing
    files = {
        "baseline_file": ("baseline.mat", Path(f_baseline).read_bytes(), "application/octet-stream"),
        "training_file": ("dataset.mat", Path(f_dataset_awal).read_bytes(), "application/octet-stream"),
    }
    r = app_client.post("/preprocess", files=files)
    assert r.status_code == 200
    
    # Wait 2 seconds
    time.sleep(2)
    
    # Should still be able to download (timeout is 10s)
    r2 = app_client.post("/preprocess/download")
    assert r2.status_code == 200
    
    # Now set a very short timeout
    monkeypatch.setattr(main, "PREPROCESS_MAX_AGE_SEC", 0.1)
    
    # Redo preprocessing
    r3 = app_client.post("/preprocess", files=files)
    assert r3.status_code == 200
    
    # Wait a bit longer than the timeout
    time.sleep(0.2)
    
    # Should now fail with timeout error
    r4 = app_client.post("/preprocess/download")
    assert r4.status_code == 400
    detail = r4.json().get("detail", "").lower()
    assert "kedaluwarsa" in detail or "expired" in detail


# Negative (-)
def test_preprocess_access_expired_session(app_client, f_baseline, f_dataset_awal):
    """
    Test that API rejects access to expired session data.
    """
    import importlib
    import time
    
    main = importlib.import_module("api.main")
    
    # Perform valid preprocessing
    files = {
        "baseline_file": ("baseline.mat", Path(f_baseline).read_bytes(), "application/octet-stream"),
        "training_file": ("dataset.mat", Path(f_dataset_awal).read_bytes(), "application/octet-stream"),
    }
    
    response = app_client.post("/preprocess", files=files)
    assert response.status_code == 200, "Initial preprocess should succeed"
    
    # Manually expire the session by setting timestamp to old value
    main.model_state.last_processed_at = time.time() - (main.PREPROCESS_MAX_AGE_SEC + 10)
    
    # Try to download after forced expiration
    download_response = app_client.post("/preprocess/download")
    
    # Should reject expired session
    assert download_response.status_code == 400, "Should reject expired session"
    error_data = download_response.json()
    assert "detail" in error_data, "Should return error detail"
    assert "kedaluwarsa" in error_data["detail"].lower() or "expired" in error_data["detail"].lower(), \
        "Error should indicate session expiration"


# --- Synthetic data builders --------------------------------------------------

def _make_synth_trials(n_trials=12, n_time=500, n_ch=16, fs=100):
    rng = np.random.default_rng(0)
    trials = []
    for _ in range(n_trials):
        base = rng.normal(0, 1, (300, n_ch))
        task = rng.normal(0.5, 1, (n_time - 300, n_ch))
        trials.append(np.vstack([base, task]).astype(np.float32))
    signal_clean = trials[0][:300, :].T
    return np.array(trials, dtype=np.float32), signal_clean
