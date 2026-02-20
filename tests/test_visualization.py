# tests/test_visualization.py
import io
import os
import tempfile
import importlib
import numpy as np
import matplotlib.pyplot as plt
import pytest


# --- util kecil untuk membuat file .h5 palsu agar lolos cek "exists" ---
def _touch_dummy_model_file(dirpath=None) -> str:
    fd, path = tempfile.mkstemp(suffix=".h5", dir=dirpath)
    try:
        os.write(fd, b"DUMMY")
    finally:
        os.close(fd)
    return path


# --- 1. /train/plot/confusion-matrix ------------------------------------------
def test_train_plot_confusion_matrix_no_results_returns_400(app_client):
    # Pastikan state training benar-benar kosong 
    main = importlib.import_module("api.main")
    main.model_state.val_cm = None
    main.model_state.val_classes = None
    main.model_state.train_cm = None
    main.model_state.train_classes = None

    r = app_client.get("/train/plot/confusion-matrix")
    assert r.status_code == 400


def test_train_plot_confusion_matrix_prefers_validation(app_client, patch_tf_light, f_preprocessed, monkeypatch):
    """
    Setelah /train, endpoint harus MEMILIH validation CM jika tersedia.
    Kita monkeypatch create_confusion_matrix_plot untuk menangkap judul.
    """
    main = importlib.import_module("api.main")

    # Latih dulu agar val_cm & train_cm ada
    files = {"dataset_file": ("preprocessed.mat", open(f_preprocessed, "rb").read(), "application/octet-stream")}
    r = app_client.post("/train", files=files)
    assert r.status_code == 200

    used_title = {"value": None}

    def fake_plot(cm, classes, title):
        used_title["value"] = title
        # balikan fig minimal
        fig, ax = plt.subplots(figsize=(2, 2))
        ax.imshow(np.asarray(cm))
        return fig

    monkeypatch.setattr(main, "create_confusion_matrix_plot", fake_plot, raising=True)

    img = app_client.get("/train/plot/confusion-matrix")
    assert img.status_code == 200
    assert img.headers["content-type"] == "image/png"
    # Harus pakai judul versi validation
    assert used_title["value"] == "Validation Confusion Matrix"


def test_train_plot_confusion_matrix_fallback_to_training(app_client, patch_tf_light, f_preprocessed):
    # Train (set keduanya), lalu kosongkan validation agar fallback ke training
    files = {"dataset_file": ("preprocessed.mat", open(f_preprocessed, "rb").read(), "application/octet-stream")}
    r = app_client.post("/train", files=files)
    assert r.status_code == 200

    main = importlib.import_module("api.main")
    main.model_state.val_cm = None
    main.model_state.val_classes = None

    img = app_client.get("/train/plot/confusion-matrix")
    assert img.status_code == 200
    assert img.headers["content-type"] == "image/png"


def test_train_plot_confusion_matrix_sets_no_cache_headers(app_client, patch_tf_light, f_preprocessed):
    # Header cache-control harus ada
    files = {"dataset_file": ("preprocessed.mat", open(f_preprocessed, "rb").read(), "application/octet-stream")}
    assert app_client.post("/train", files=files).status_code == 200

    resp = app_client.get("/train/plot/confusion-matrix")
    assert resp.status_code == 200
    assert "Cache-Control" in resp.headers
    assert "no-cache" in resp.headers["Cache-Control"]

