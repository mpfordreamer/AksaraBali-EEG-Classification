import pytest
from fastapi import status

# test_main.py
def test_app_health(app_client):
    r = app_client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
