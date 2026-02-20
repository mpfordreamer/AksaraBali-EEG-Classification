"""
Configuration settings for Aksara Bali EEG Load Tests
"""
import os

# ====== API CONFIGURATION ======
# Base URL of the API to test
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8001")

# Timeout for API requests (ML operations can take time)
API_TIMEOUT = int(os.getenv("API_TIMEOUT", "180"))  # seconds

# ====== LOAD TEST PARAMETERS ======
# Wait times between user actions (milliseconds)
MIN_WAIT = int(os.getenv("MIN_WAIT", "1000"))  # 1 second
MAX_WAIT = int(os.getenv("MAX_WAIT", "5000"))  # 5 seconds

# For training operations (longer waits)
TRAIN_MIN_WAIT = int(os.getenv("TRAIN_MIN_WAIT", "5000"))  # 5 seconds
TRAIN_MAX_WAIT = int(os.getenv("TRAIN_MAX_WAIT", "15000"))  # 15 seconds

# ====== FILE PATHS ======
# Directory containing test data files
TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "test_data")

# Test data files
TEST_FILES = {
    "baseline": os.path.join(TEST_DATA_DIR, "baseline.mat"),
    "dataset_awal": os.path.join(TEST_DATA_DIR, "dataset_awal.mat"),
    "preprocessed": os.path.join(TEST_DATA_DIR, "preprocessed.mat"),
    "predict": os.path.join(TEST_DATA_DIR, "predict.mat"),
}

# ====== PERFORMANCE THRESHOLDS ======
# Maximum acceptable response times (milliseconds)
THRESHOLDS = {
    "health_max_response": 100,        # Health check should be very fast
    "preprocess_max_response": 10000,  # 10 seconds for preprocessing
    "train_max_response": 180000,      # 3 minutes for training
    "predict_max_response": 5000,      # 5 seconds for prediction
    "models_list_max_response": 500,   # 500ms for listing models
}

# ====== USER WEIGHTS ======
# Distribution of user types in mixed workload scenarios
USER_WEIGHTS = {
    "HealthCheckUser": 2,      # Frequent health checks
    "PreprocessingUser": 1,    # Occasional preprocessing
    "TrainingUser": 1,         # Occasional training
    "PredictionUser": 3,       # Most common operation
    "MixedWorkflowUser": 2,    # Some users do complete workflows
}

# ====== REPORT SETTINGS ======
# Output directory for reports
REPORT_DIR = os.path.join(os.path.dirname(__file__), "reports")
os.makedirs(REPORT_DIR, exist_ok=True)

# Default report file names
HTML_REPORT = os.path.join(REPORT_DIR, "load_test_report.html")
CSV_STATS_REPORT = os.path.join(REPORT_DIR, "load_test_stats.csv")
CSV_FAILURES_REPORT = os.path.join(REPORT_DIR, "load_test_failures.csv")

# ====== VALIDATION SETTINGS ======
# Enable/disable response validation
VALIDATE_RESPONSES = os.getenv("VALIDATE_RESPONSES", "true").lower() == "true"

# Expected response structures for validation
EXPECTED_RESPONSES = {
    "preprocess": ["message", "trial_features_shape", "base_features_shape"],
    "train": ["val_accuracy", "val_loss", "precision", "recall", "f1", "model_path"],
    "predict": ["message", "predicted_labels", "predicted_probs", "num_trials"],
    "models_list": ["models", "default_model"],
}

# ====== LOGGING ======
# Log level for load test execution
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Enable detailed logging of requests/responses
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"
