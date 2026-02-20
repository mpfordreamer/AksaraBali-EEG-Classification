# tests/conftest.py
import io
import os
import sys
from pathlib import Path
import importlib
import numpy as np
import scipy.io as sio
import pytest
import warnings

# Make sure the project root is importable (…/Deployment)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Silence the TF Lite → JAX deprecation message across all tests
warnings.filterwarnings(
    "ignore",
    message=r"jax\.xla_computation is deprecated.*",
    category=DeprecationWarning,
)

# Silence httpx TestClient deprecation warnings
warnings.filterwarnings(
    "ignore",
    message=r"The 'app' shortcut is now deprecated.*",
    category=DeprecationWarning,
)

# Stub streamlit early (some environments may not have it)
try:
    import streamlit  # noqa: F401
except Exception:
    import types
    st = types.SimpleNamespace(
        set_page_config=lambda *a, **k: None,
        title=lambda *a, **k: None,
        write=lambda *a, **k: None,
        success=lambda *a, **k: None,
        caption=lambda *a, **k: None,
        button=lambda *a, **k: False,
        # simple context manager for "with col:"
        columns=lambda n: [types.SimpleNamespace(__enter__=lambda s: s, __exit__=lambda *a, **k: False) for _ in range(n)],
        session_state={}
    )
    sys.modules["streamlit"] = st

# Import app factory and state class from your API package
from fastapi.testclient import TestClient
from api.main import create_app, ModelState  # requires you added create_app() in api/main.py

# Make `import main` (legacy) resolve to `api.main` to keep old tests working
if "main" not in sys.modules:
    sys.modules["main"] = importlib.import_module("api.main")


# =======================
# Core client fixture (factory-based)
# =======================
@pytest.fixture
def app_client(tmp_path, monkeypatch):
    """
    Build a brand-new FastAPI app per test using the factory.
    Models directory and default model are isolated to tmp_path.
    """
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Speed up any training that reads env (if your code uses these)
    monkeypatch.setenv("EPOCHS", "1")
    monkeypatch.setenv("VAL_PATIENCE", "1")

    # Point the app to the temp models location
    monkeypatch.setenv("MODELS_DIR", str(models_dir))
    monkeypatch.setenv("DEFAULT_MODEL_PATH", str(models_dir / "LSTM_Model_Final.h5"))

    # Ensure Streamlit launcher never runs in tests
    monkeypatch.delenv("RUN_STREAMLIT", raising=False)

    app = create_app()                  # ← brand new app instance
    app.state.model_state = ModelState()  # ← fresh state (extra safety)

    with TestClient(app) as c:
        yield c


# =======================
# Test data path fixtures
# =======================
@pytest.fixture(scope="session")
def data_dir() -> Path:
    return Path(__file__).parent / "test_data"

@pytest.fixture(scope="session")
def f_baseline(data_dir): return data_dir / "baseline.mat"

@pytest.fixture(scope="session")
def f_dataset_awal(data_dir): return data_dir / "dataset_awal.mat"

@pytest.fixture(scope="session")
def f_preprocessed(data_dir): return data_dir / "preprocessed.mat"

@pytest.fixture(scope="session")
def f_predict(data_dir): return data_dir / "predict.mat"

@pytest.fixture(scope="session")
def f_predict1class(data_dir): return data_dir / "predict1class.mat"

@pytest.fixture(scope="session")
def f_invalid(data_dir): return data_dir / "invalid.mat"


# =======================
# Small util to create .mat bytes on the fly
# =======================
@pytest.fixture()
def make_mat_bytes():
    def _make(dic):
        buf = io.BytesIO()
        sio.savemat(buf, dic)
        return buf.getvalue()
    return _make


# =======================
# Fast Keras/TensorFlow dummies for speedy tests
# =======================
class _DummyHistory:
    def __init__(self):
        self.history = {
            "loss": [0.8, 0.6],
            "val_loss": [0.9, 0.7],
            "accuracy": [0.5, 0.7],
            "val_accuracy": [0.45, 0.65],
        }

class _DummyEarlyStopping:
    def __init__(self, monitor='val_loss', patience=10, restore_best_weights=True):
        self.stopped_epoch = 1

class _DummyModel:
    def __init__(self):
        self._num_classes = 6
    def add(self, *a, **k): pass
    def compile(self, *a, **k): pass
    def fit(self, X, y, validation_data=None, epochs=100, batch_size=8, callbacks=None, verbose=0):
        if y is not None and y.ndim == 2:
            self._num_classes = y.shape[1]
        return _DummyHistory()
    def predict(self, X, verbose=0):
        rng = np.random.RandomState(123)
        logits = rng.rand(X.shape[0], self._num_classes)
        return logits / logits.sum(axis=1, keepdims=True)
    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(b"DUMMYMODEL")

@pytest.fixture()
def patch_tf_light(monkeypatch):
    """
    Patch Keras symbols used by the app to fast dummies so /train runs quickly.
    """
    main = importlib.import_module("api.main")
    # Expose also via legacy alias if some tests import "main"
    sys.modules["main"] = main

    class _Layer:
        def __init__(self, *a, **k): pass

    monkeypatch.setattr(main, "Sequential", _DummyModel, raising=True)
    monkeypatch.setattr(main, "Input", _Layer, raising=True)
    monkeypatch.setattr(main, "LSTM", _Layer, raising=True)
    monkeypatch.setattr(main, "Dense", _Layer, raising=True)
    monkeypatch.setattr(main, "Dropout", _Layer, raising=True)
    monkeypatch.setattr(main, "EarlyStopping", _DummyEarlyStopping, raising=True)

    # simple to_categorical replacement
    monkeypatch.setattr(
        main, "to_categorical",
        lambda y: np.eye(int(np.max(y)) + 1, dtype=np.float32)[y],
        raising=True
    )
    yield

def pytest_terminal_summary(terminalreporter, exitstatus, config):
    passed = [rep for rep in terminalreporter.stats.get('passed', [])]
    if passed:
        terminalreporter.write('\n' + '='*40 + '\n')
        terminalreporter.write('PASSED TESTS:\n')
        for rep in passed:
            terminalreporter.write(f'  {rep.nodeid}\n')
        terminalreporter.write('='*40 + '\n')
